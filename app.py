#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A股ETF多项式回归分析 - 基于统计学的自动阶数选择
支持自动识别市场后缀（.SH / .SZ）
采用序贯F检验 + BIC最小化 + 交叉验证(1-SE法则)
固定预测未来7个交易日
"""

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from datetime import datetime, timedelta
from tickflow import TickFlow
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# ---------- 页面设置 ----------
st.set_page_config(page_title="A股ETF多项式回归分析", layout="wide")
st.title("📈 A股ETF多项式回归分析与预测")
st.markdown("""
**自动选择机制**：
1. **序贯 F 检验**（新增项 p<0.05）
2. **BIC 最小化**（惩罚过拟合）
3. **交叉验证 (1-SE 法则)**（确保泛化能力）
""")

# ---------- 侧边栏 ----------
st.sidebar.header("参数设置")
default_symbol = st.sidebar.text_input("ETF代码（输入数字即可，如 515050）", value="515050")
analyze_btn = st.sidebar.button("开始分析", type="primary")
st.sidebar.info("预测天数固定为 7 天\n阶数自动在 1~7 中优选（统计学准则）")
st.sidebar.caption("💡 系统自动识别上海（.SH）或深圳（.SZ）市场")

# ---------- 辅助函数：自动补全市场后缀 ----------
def normalize_symbol(symbol):
    """自动补全市场后缀（用户无需输入 .SH 或 .SZ）"""
    symbol = symbol.strip().upper()
    if '.' in symbol:
        return symbol
    if symbol.startswith('6'):
        return f"{symbol}.SH"
    elif symbol.startswith('0') or symbol.startswith('3'):
        return f"{symbol}.SZ"
    else:
        return f"{symbol}.SH"   # 默认上海

# ---------- 数据获取 ----------
@st.cache_data(show_spinner=False)
def get_data(symbol):
    tf = TickFlow.free()
    df = tf.klines.get(symbol, period="1d", count=5000, as_dataframe=True)
    if df is None or df.empty:
        raise Exception("未获取到数据，请检查代码或网络")
    df = df[['trade_date', 'close']].copy()
    df.columns = ['date', 'close']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

# ---------- 核心建模函数 ----------
def build_model_stats(data, predict_days, order):
    """返回包含 AIC/BIC/F-test 结果的模型字典"""
    n = len(data)
    X_seq = np.arange(1, n + 1).reshape(-1, 1)
    X_poly = np.column_stack([X_seq ** i for i in range(1, order + 1)])
    X_poly_const = sm.add_constant(X_poly)
    Y = data['close'].values
    
    model = sm.OLS(Y, X_poly_const)
    results = model.fit()
    
    params = results.params
    pvals = results.pvalues
    coeffs = {f'coef_{i}': p for i, p in enumerate(params)}
    p_values = {f'p_{i}': pv for i, pv in enumerate(pvals)}
    
    future_X = np.arange(n + 1, n + predict_days + 1).reshape(-1, 1)
    future_X_poly = np.column_stack([future_X ** i for i in range(1, order + 1)])
    future_X_poly_const = sm.add_constant(future_X_poly)
    pred = results.get_prediction(future_X_poly_const)
    pred_summary = pred.summary_frame(alpha=0.05)
    
    return {
        'results': results,
        'coeffs': coeffs,
        'p_values': p_values,
        'r2': results.rsquared,
        'adj_r2': results.rsquared_adj,
        'aic': results.aic,
        'bic': results.bic,
        'f_pvalue': results.f_pvalue,
        'fitted': results.fittedvalues,
        'n': n,
        'order': order,
        'pred_values': pred_summary['mean'].values,
        'ci_lower': pred_summary['obs_ci_lower'].values,
        'ci_upper': pred_summary['obs_ci_upper'].values,
        'future_X': future_X.flatten(),
        'all_p_significant': all(pv < 0.05 for pv in pvals),
        'mse': np.mean(results.resid ** 2)
    }

# ---------- 交叉验证（5折，计算MSE） ----------
def get_cv_mse(data, order):
    X = np.arange(1, len(data) + 1).reshape(-1, 1)
    y = data['close'].values
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=order, include_bias=False)),
        ('linear', LinearRegression())
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    return -scores.mean()

# ---------- 主选择函数（统计准则） ----------
def select_best_model_statistical(data, predict_days):
    max_order = 7
    candidates = []
    cv_mse_dict = {}
    
    for order in range(1, max_order + 1):
        try:
            model_dict = build_model_stats(data, predict_days, order)
            cv_mse = get_cv_mse(data, order)
            cv_mse_dict[order] = cv_mse
            model_dict['cv_mse'] = cv_mse
            candidates.append(model_dict)
        except Exception as e:
            st.warning(f"阶数 {order} 建模失败: {e}")
            continue
    
    if not candidates:
        raise Exception("所有阶数建模均失败。")
    
    sig_models = [m for m in candidates if m['all_p_significant']]
    
    if sig_models:
        bic_min_model = min(sig_models, key=lambda m: m['bic'])
        cv_min_model = min(sig_models, key=lambda m: m['cv_mse'])
        
        if bic_min_model['order'] != cv_min_model['order']:
            best = bic_min_model if bic_min_model['order'] < cv_min_model['order'] else cv_min_model
            best['selection_note'] = (
                f"📊 综合选择 {best['order']} 阶\n"
                f"   - BIC 最优为 {bic_min_model['order']} 阶 (BIC={bic_min_model['bic']:.2f})\n"
                f"   - CV 最优为 {cv_min_model['order']} 阶 (MSE={cv_min_model['cv_mse']:.6f})\n"
                f"   - 根据节俭原则，选取较低阶数"
            )
        else:
            best = bic_min_model
            best['selection_note'] = (
                f"📊 选择 {best['order']} 阶（BIC 与 CV 同时最优）\n"
                f"   BIC={best['bic']:.2f}, CV-MSE={best['cv_mse']:.6f}"
            )
    else:
        best = min(candidates, key=lambda m: m['bic'])
        best['selection_note'] = (
            f"⚠️ 所有阶数均存在 p≥0.05 的系数，\n"
            f"   选择 BIC 最小的 {best['order']} 阶（BIC={best['bic']:.2f}）"
        )
    
    best['all_orders_summary'] = pd.DataFrame({
        '阶数': [m['order'] for m in candidates],
        '调整R²': [m['adj_r2'] for m in candidates],
        'AIC': [m['aic'] for m in candidates],
        'BIC': [m['bic'] for m in candidates],
        'CV-MSE': [m['cv_mse'] for m in candidates],
        '全系数显著?': ['✅' if m['all_p_significant'] else '❌' for m in candidates]
    }).set_index('阶数')
    
    return best

# ---------- 绘图函数 ----------
def create_figure(data, model_dict, symbol, predict_days):
    dates = data['date']
    actual = data['close']
    fitted = model_dict['fitted']
    pred_values = model_dict['pred_values']
    ci_lower = model_dict['ci_lower']
    ci_upper = model_dict['ci_upper']
    order = model_dict['order']
    
    last_date = dates.iloc[-1]
    pred_dates = [last_date + timedelta(days=i+1) for i in range(len(pred_values))]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name='实际收盘价',
                             line=dict(color='#1f77b4', width=2.5),
                             hovertemplate='<b>%{x|%Y/%m/%d}</b><br>收盘价: %{y:.4f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=dates, y=fitted, mode='lines', name='模型拟合值',
                             line=dict(color='#d62728', width=2, dash='dash'),
                             hovertemplate='<b>%{x|%Y/%m/%d}</b><br>拟合值: %{y:.4f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=pred_dates, y=pred_values, mode='lines+markers',
                             name='预测 (t+1 ~ t+7)',
                             line=dict(color='#2ca02c', width=2, dash='dot'),
                             marker=dict(color='#2ca02c', size=8),
                             hovertemplate='<b>%{x|%Y/%m/%d}</b><br>预测值: %{y:.4f}<extra></extra>'))
    
    for i, (x, y, lo, hi) in enumerate(zip(pred_dates, pred_values, ci_lower, ci_upper)):
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', name='95% CI' if i==0 else None,
                                 marker=dict(size=0),
                                 error_y=dict(type='data', symmetric=False,
                                              array=[hi-y], arrayminus=[y-lo],
                                              color='#2ca02c', thickness=2, width=8)))
    
    fig.add_trace(go.Scatter(x=pred_dates + pred_dates[::-1],
                             y=list(ci_upper) + list(ci_lower[::-1]),
                             fill='toself', fillcolor='rgba(44,160,44,0.15)',
                             line=dict(color='rgba(255,255,255,0)'),
                             name='95% CI（带状）', hoverinfo='skip'))
    
    fig.update_layout(title=f"{symbol} {order} 阶多项式回归（未来7天预测）",
                      xaxis=dict(title="日期", tickformat='%Y/%m/%d', rangeslider=dict(visible=True),
                                 rangeselector=dict(buttons=[dict(count=1, label="1月", step="month", stepmode="backward"),
                                                             dict(count=3, label="3月", step="month", stepmode="backward"),
                                                             dict(count=6, label="6月", step="month", stepmode="backward"),
                                                             dict(count=1, label="1年", step="year", stepmode="backward"),
                                                             dict(count=3, label="3年", step="year", stepmode="backward"),
                                                             dict(count=5, label="5年", step="year", stepmode="backward"),
                                                             dict(step="all", label="全部")])),
                      yaxis=dict(title="收盘价", tickformat='.2f'),
                      hovermode='x unified', height=600)
    
    all_dates = list(dates) + pred_dates
    fig.update_xaxes(range=[all_dates[0], all_dates[-1] + timedelta(days=2)])
    fig.update_yaxes(range=[min(actual.min(), ci_lower.min())*0.95,
                             max(actual.max(), ci_upper.max())*1.05])
    return fig

# ---------- 辅助：公式生成 ----------
def generate_formula(model_dict):
    coeffs = model_dict['coeffs']
    terms = [f"{coeffs['coef_0']:.6f}"]
    for i in range(1, model_dict['order'] + 1):
        coef = coeffs[f'coef_{i}']
        if i == 1:
            terms.append(f"{coef:.6f}·X")
        else:
            terms.append(f"{coef:.6f}·X^{i}")
    return "Y = " + " + ".join(terms)

# ---------- 主程序 ----------
if analyze_btn:
    raw_symbol = default_symbol.strip()
    symbol = normalize_symbol(raw_symbol)  # 自动补全后缀
    predict_days = 7
    
    with st.spinner(f"正在获取 {symbol} 数据并基于统计学准则（F检验+BIC+CV）选择最优阶数（1~7阶）..."):
        try:
            data = get_data(symbol)
        except Exception as e:
            st.error(f"数据获取失败: {e}")
            st.stop()
        
        try:
            best = select_best_model_statistical(data, predict_days)
        except Exception as e:
            st.error(f"模型选择失败: {e}")
            st.stop()
        
        # 展示选择说明
        st.info(best['selection_note'])
        
        # 展示各阶数对比表
        with st.expander("📋 查看 1~7 阶详细对比数据（调整R²、AIC、BIC、CV-MSE）"):
            st.dataframe(best['all_orders_summary'], use_container_width=True)
        
        # 数据概览
        st.subheader("📊 数据概览")
        col1, col2, col3 = st.columns(3)
        col1.metric("数据点数", best['n'])
        col2.metric("起始日期", data['date'].min().strftime('%Y/%m/%d'))
        col3.metric("最新日期", data['date'].max().strftime('%Y/%m/%d'))
        
        # 模型摘要
        st.subheader("📐 回归模型摘要")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**R²** = {best['r2']:.6f}")
            st.markdown(f"**调整 R²** = {best['adj_r2']:.6f}")
            st.markdown(f"**F 检验 p 值** = {best['f_pvalue']:.2e}")
            st.markdown(f"**AIC** = {best['aic']:.2f}")
            st.markdown(f"**BIC** = {best['bic']:.2f}")
        with col2:
            st.markdown("**回归系数及 p 值**")
            order = best['order']
            index_names = ['常数项'] + [f"X^{i}" if i>1 else "X" for i in range(1, order+1)]
            coef_df = pd.DataFrame({
                '系数': [best['coeffs'][f'coef_{i}'] for i in range(order+1)],
                'p 值': [best['p_values'][f'p_{i}'] for i in range(order+1)]
            }, index=index_names)
            st.dataframe(coef_df, column_config={"p 值": st.column_config.NumberColumn(format="%.2f")},
                         use_container_width=True)
        
        # 预测结果
        st.subheader("🔮 未来 7 个交易日预测 (95% 置信区间)")
        pred_df = pd.DataFrame({
            '交易日': [f"t+{i+1}" for i in range(7)],
            '序列号': best['future_X'],
            '预测收盘价': best['pred_values'],
            '下限 (95% CI)': best['ci_lower'],
            '上限 (95% CI)': best['ci_upper']
        })
        st.dataframe(pred_df, use_container_width=True)
        
        # 图表
        st.subheader("📈 价格走势与预测图")
        fig = create_figure(data, best, symbol, predict_days)
        st.plotly_chart(fig, use_container_width=True)
        
        # 公式
        st.subheader("📝 模型公式")
        st.latex(generate_formula(best))

else:
    st.info("👈 在左侧输入 ETF 代码（纯数字），点击『开始分析』按钮查看基于统计学准则自动选阶的结果。")