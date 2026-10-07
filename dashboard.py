import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st
from scipy import stats

st.set_page_config(page_title='Imóveis em Ames', page_icon='🏠', layout='wide')

AZUL = '#2a78d6'
VERMELHO = '#e34948'

ROTULOS = {
    'preco': 'Preço (US$)',
    'preco_m2': 'Preço por m² (US$)',
    'area_total': 'Área total (m²)',
    'area_util': 'Área útil (m²)',
    'area_lote': 'Área do lote (m²)',
    'qualidade': 'Qualidade (1 a 10)',
    'conservacao': 'Conservação (1 a 10)',
    'idade': 'Idade (anos)',
    'banheiros': 'Banheiros',
    'quartos': 'Quartos',
    'vagas_garagem': 'Vagas de garagem',
    'lareiras': 'Lareiras',
    'ar_central': 'Ar central',
    'tem_lareira': 'Tem lareira',
    'reformada': 'Reformada',
    'tipo_imovel': 'Tipo de imóvel',
    'tipo_garagem': 'Tipo de garagem',
    'bairro': 'Bairro',
    'faixa_preco': 'Faixa de preço',
    'periodo': 'Período',
    'ano_venda': 'Ano da venda',
    'mes_venda': 'Mês da venda',
}
NUMERICAS = ['preco', 'preco_m2', 'area_total', 'area_util', 'area_lote', 'qualidade', 'conservacao', 'idade',
             'banheiros', 'quartos', 'vagas_garagem', 'lareiras']
CATEGORICAS = ['ar_central', 'tem_lareira', 'reformada', 'tipo_imovel', 'tipo_garagem', 'periodo', 'faixa_preco',
               'ano_venda', 'bairro']
PREDITORES = ['area_total', 'qualidade', 'conservacao', 'idade', 'vagas_garagem', 'lareiras']


def rotulo(coluna):
    return ROTULOS[coluna]


def texto_p(p):
    return '< 0,0001' if p < 0.0001 else f'{p:.4f}'


def ic_media(valores, confianca=0.95):
    valores = np.asarray(valores, dtype=float)
    media = valores.mean()
    margem = stats.t.ppf((1 + confianca) / 2, len(valores) - 1) * valores.std(ddof=1) / np.sqrt(len(valores))
    return media, media - margem, media + margem


def grafico_medias(tabela, eixo_x, eixo_y, titulo):
    fig = go.Figure(go.Scatter(
        x=tabela.index.astype(str), y=tabela['media'], mode='markers', marker=dict(size=11, color=AZUL),
        error_y=dict(type='data', symmetric=False, array=tabela['superior'] - tabela['media'],
                     arrayminus=tabela['media'] - tabela['inferior'], color='gray'),
        hovertemplate='%{x}<br>Média: %{y:,.1f}<extra></extra>'))
    fig.update_layout(title=titulo, xaxis_title=eixo_x, yaxis_title=eixo_y, xaxis_type='category', height=420)
    return fig


@st.cache_data
def carregar():
    return pd.read_csv('dados_tratados.csv')


dados = carregar()

st.sidebar.header('Filtros')
anos = st.sidebar.slider('Ano da venda', 2006, 2010, (2006, 2010))
tipos = st.sidebar.multiselect('Tipo de imóvel', sorted(dados['tipo_imovel'].unique()), placeholder='Todos')
bairros = st.sidebar.multiselect('Bairro', sorted(dados['bairro'].unique()), placeholder='Todos')
notas = st.sidebar.slider('Qualidade', 1, 10, (1, 10))
confianca = st.sidebar.select_slider('Nível de confiança', [0.90, 0.95, 0.99], value=0.95, format_func=lambda v: f'{v:.0%}')
alfa = 1 - confianca

df = dados[dados['ano_venda'].between(*anos) & dados['qualidade'].between(*notas)]
if tipos:
    df = df[df['tipo_imovel'].isin(tipos)]
if bairros:
    df = df[df['bairro'].isin(bairros)]
st.sidebar.metric('Casas selecionadas', len(df))

st.title('Preço de imóveis em Ames (EUA), 2006 a 2010')

if len(df) < 30:
    st.warning('Menos de 30 casas na seleção. Amplie os filtros.')
    st.stop()

aba_geral, aba_relacoes, aba_local, aba_testes, aba_regressao = st.tabs(
    ['Visão geral', 'Relações', 'Bairros e tempo', 'Testes de hipóteses', 'Regressão e previsão'])

with aba_geral:
    media, inferior, superior = ic_media(df['preco'], confianca)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Casas', len(df))
    c2.metric('Preço médio (US$)', f'{media:,.0f}', f'IC {confianca:.0%}: {inferior:,.0f} a {superior:,.0f}', delta_color='off')
    c3.metric('Preço mediano (US$)', f'{df["preco"].median():,.0f}')
    c4.metric('Preço médio por m² (US$)', f'{df["preco_m2"].mean():,.0f}')

    esquerda, direita = st.columns(2)
    with esquerda:
        variavel = st.selectbox('Variável', NUMERICAS, format_func=rotulo)
        fig = px.histogram(df, x=variavel, nbins=40, labels=ROTULOS, color_discrete_sequence=[AZUL], title=f'Distribuição: {rotulo(variavel)}')
        fig.add_vline(df[variavel].mean(), line_dash='dash', line_color=VERMELHO, annotation_text='média')
        fig.add_vline(df[variavel].median(), line_dash='dot', line_color='gray', annotation_text='mediana', annotation_position='top left')
        fig.update_layout(yaxis_title='Número de casas', height=420)
        st.plotly_chart(fig, width='stretch')
    with direita:
        grupo = st.selectbox('Comparar preço por', CATEGORICAS, format_func=rotulo)
        ordem = df.groupby(grupo)['preco'].median().sort_values().index.tolist()
        fig = px.box(df, x=grupo, y='preco', labels=ROTULOS, category_orders={grupo: ordem}, color_discrete_sequence=[AZUL],
                     title=f'Preço por {rotulo(grupo).lower()}')
        fig.update_layout(height=420, xaxis_type='category')
        st.plotly_chart(fig, width='stretch')

    resumo = df[NUMERICAS].describe().T[['mean', '50%', 'std', 'min', 'max']]
    resumo.columns = ['Média', 'Mediana', 'Desvio padrão', 'Mínimo', 'Máximo']
    resumo.index = [rotulo(c) for c in resumo.index]
    st.dataframe(resumo.style.format('{:,.1f}'), width='stretch')

with aba_relacoes:
    c1, c2, c3 = st.columns(3)
    eixo_x = c1.selectbox('Eixo X', NUMERICAS, index=NUMERICAS.index('area_total'), format_func=rotulo)
    eixo_y = c2.selectbox('Eixo Y', NUMERICAS, format_func=rotulo)
    reta = c3.checkbox('Mostrar reta de regressão', value=True)
    fig = px.scatter(df, x=eixo_x, y=eixo_y, opacity=0.5, labels=ROTULOS, color_discrete_sequence=[AZUL],
                     trendline='ols' if reta else None, trendline_color_override=VERMELHO,
                     hover_data=['bairro', 'tipo_imovel'], title=f'{rotulo(eixo_y)} × {rotulo(eixo_x)}')
    fig.update_layout(height=480)
    st.plotly_chart(fig, width='stretch')

    if eixo_x != eixo_y:
        correlacao = stats.pearsonr(df[eixo_x], df[eixo_y])
        ic = correlacao.confidence_interval(confianca)
        simples = smf.ols(f'{eixo_y} ~ {eixo_x}', data=df).fit()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Correlação de Pearson (r)', f'{correlacao.statistic:.3f}', f'IC {confianca:.0%}: {ic.low:.3f} a {ic.high:.3f}', delta_color='off')
        c2.metric('p-valor da correlação', texto_p(correlacao.pvalue))
        c3.metric('Inclinação da reta', f'{simples.params[eixo_x]:,.2f}')
        c4.metric('R² da regressão simples', f'{simples.rsquared:.3f}')

    matriz = df[NUMERICAS].corr()
    matriz.index = matriz.columns = [rotulo(c) for c in NUMERICAS]
    fig = px.imshow(matriz, text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale=[AZUL, '#f0efec', VERMELHO],
                    aspect='auto', title='Matriz de correlação de Pearson')
    fig.update_layout(height=560)
    st.plotly_chart(fig, width='stretch')

with aba_local:
    metrica = st.radio('Métrica', ['preco_m2', 'preco'], format_func=rotulo, horizontal=True)
    minimo = st.slider('Mínimo de vendas por bairro', 2, 50, 10)

    por_bairro = df.groupby('bairro')[metrica].agg(['count', 'mean', 'std'])
    por_bairro = por_bairro[por_bairro['count'] >= minimo].sort_values('mean')
    margem = stats.t.ppf((1 + confianca) / 2, por_bairro['count'] - 1) * por_bairro['std'] / np.sqrt(por_bairro['count'])
    fig = go.Figure(go.Scatter(
        x=por_bairro['mean'], y=por_bairro.index, mode='markers', marker=dict(size=10, color=AZUL),
        error_x=dict(type='data', array=margem, color='gray'), customdata=por_bairro['count'],
        hovertemplate='<b>%{y}</b><br>Média: %{x:,.0f}<br>Vendas: %{customdata}<extra></extra>'))
    fig.add_vline(df[metrica].mean(), line_dash='dash', line_color=VERMELHO, annotation_text='média da seleção')
    fig.update_layout(title=f'{rotulo(metrica)}: média e IC de {confianca:.0%} por bairro', xaxis_title=rotulo(metrica),
                      height=max(400, 24 * len(por_bairro) + 120))
    st.plotly_chart(fig, width='stretch')

    esquerda, direita = st.columns(2)
    with esquerda:
        vendas = df.groupby('mes_venda').size().reset_index(name='vendas')
        fig = px.bar(vendas, x='mes_venda', y='vendas', labels={'mes_venda': 'Mês', 'vendas': 'Número de vendas'},
                     color_discrete_sequence=[AZUL], title='Número de vendas por mês')
        fig.update_layout(xaxis_type='category', height=400)
        st.plotly_chart(fig, width='stretch')
    with direita:
        por_ano = df.groupby('ano_venda')[metrica].apply(lambda v: pd.Series(ic_media(v, confianca), index=['media', 'inferior', 'superior'])).unstack()
        st.plotly_chart(grafico_medias(por_ano, 'Ano da venda', rotulo(metrica), f'{rotulo(metrica)}: média e IC de {confianca:.0%} por ano'), width='stretch')

with aba_testes:
    st.caption(f'Nível de significância: α = {alfa:.2f}')
    tipo_teste = st.radio('Teste', ['Comparação de médias (teste t / ANOVA)', 'Associação entre categorias (qui-quadrado)'], horizontal=True)

    if tipo_teste.startswith('Comparação'):
        c1, c2 = st.columns(2)
        resposta = c1.selectbox('Variável numérica', NUMERICAS, format_func=rotulo, key='teste_resposta')
        fator = c2.selectbox('Grupos definidos por', [c for c in CATEGORICAS if c != 'faixa_preco'], format_func=rotulo, key='teste_fator')
        contagem = df[fator].value_counts()
        niveis = sorted(contagem[contagem >= 5].index.tolist(), key=str)
        escolhidos = st.multiselect('Grupos comparados', niveis, default=niveis)

        if len(escolhidos) < 2:
            st.info('Selecione pelo menos 2 grupos.')
        else:
            parte = df[df[fator].isin(escolhidos)]
            amostras = [parte.loc[parte[fator] == nivel, resposta] for nivel in escolhidos]
            medias = pd.DataFrame([ic_media(a, confianca) for a in amostras], index=escolhidos, columns=['media', 'inferior', 'superior'])

            esquerda, direita = st.columns(2)
            fig = px.box(parte, x=fator, y=resposta, labels=ROTULOS, category_orders={fator: escolhidos},
                         color_discrete_sequence=[AZUL], title='Distribuição por grupo')
            fig.update_layout(height=420, xaxis_type='category')
            esquerda.plotly_chart(fig, width='stretch')
            direita.plotly_chart(grafico_medias(medias, rotulo(fator), rotulo(resposta), f'Média e IC de {confianca:.0%} por grupo'), width='stretch')

            descricao = pd.DataFrame({'Grupo': [str(e) for e in escolhidos], 'n': [len(a) for a in amostras],
                                      'Média': [a.mean() for a in amostras], 'Desvio padrão': [a.std() for a in amostras]})
            st.dataframe(descricao.style.format({'Média': '{:,.1f}', 'Desvio padrão': '{:,.1f}'}), hide_index=True, width='stretch')

            levene = stats.levene(*amostras)
            if len(escolhidos) == 2:
                teste = stats.ttest_ind(amostras[0], amostras[1], equal_var=False)
                ic = teste.confidence_interval(confianca)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('Teste t de Welch: t', f'{teste.statistic:.2f}')
                c2.metric('Graus de liberdade', f'{teste.df:.0f}')
                c3.metric('p-valor', texto_p(teste.pvalue))
                c4.metric('Levene (variâncias): p-valor', texto_p(levene.pvalue))
                st.metric(f'Diferença das médias ({escolhidos[0]} − {escolhidos[1]})', f'{amostras[0].mean() - amostras[1].mean():,.1f}',
                          f'IC {confianca:.0%}: {ic.low:,.1f} a {ic.high:,.1f}', delta_color='off')
            else:
                anova = stats.f_oneway(*amostras)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('ANOVA: F', f'{anova.statistic:.2f}')
                c2.metric('Graus de liberdade', f'{len(escolhidos) - 1} e {len(parte) - len(escolhidos)}')
                c3.metric('p-valor', texto_p(anova.pvalue))
                c4.metric('Levene (variâncias): p-valor', texto_p(levene.pvalue))
    else:
        opcoes = ['tem_lareira', 'ar_central', 'reformada', 'faixa_preco', 'tipo_imovel', 'tipo_garagem', 'periodo']
        c1, c2 = st.columns(2)
        linha = c1.selectbox('Variável nas linhas', opcoes, format_func=rotulo)
        coluna = c2.selectbox('Variável nas colunas', opcoes, index=3, format_func=rotulo)
        if linha == coluna:
            st.info('Escolha duas variáveis diferentes.')
        else:
            tabela = pd.crosstab(df[linha], df[coluna])
            qui2, p_valor, gl, esperado = stats.chi2_contingency(tabela)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Qui-quadrado', f'{qui2:,.1f}')
            c2.metric('Graus de liberdade', gl)
            c3.metric('p-valor', texto_p(p_valor))
            c4.metric('Menor frequência esperada', f'{esperado.min():.1f}')
            exibir = st.radio('Mostrar', ['Percentual por linha', 'Contagem observada', 'Contagem esperada'], horizontal=True)
            if exibir == 'Percentual por linha':
                valores, formato = tabela.div(tabela.sum(axis=1), axis=0) * 100, '.1f'
            elif exibir == 'Contagem observada':
                valores, formato = tabela, 'd'
            else:
                valores, formato = pd.DataFrame(esperado, index=tabela.index, columns=tabela.columns), '.0f'
            fig = px.imshow(valores, text_auto=formato, aspect='auto', color_continuous_scale=['#cde2fb', '#3987e5', '#0d366b'],
                            labels=dict(x=rotulo(coluna), y=rotulo(linha), color=exibir), title=f'Tabela de contingência: {exibir.lower()}')
            fig.update_layout(height=380 + 25 * len(tabela))
            st.plotly_chart(fig, width='stretch')

with aba_regressao:
    escolhidas = st.multiselect('Variáveis explicativas', PREDITORES + ['banheiros', 'quartos', 'area_lote'], default=PREDITORES, format_func=rotulo)
    treino = df[df['ano_venda'] <= 2009]
    teste = df[df['ano_venda'] == 2010]

    if not escolhidas:
        st.info('Selecione pelo menos uma variável explicativa.')
    elif len(treino) < 50 or len(teste) < 10:
        st.info('O modelo treina com vendas de 2006 a 2009 e testa em 2010. Inclua esses anos nos filtros.')
    else:
        modelo = smf.ols('preco ~ ' + ' + '.join(escolhidas), data=treino).fit()
        previsto = modelo.predict(teste)
        erro = teste['preco'] - previsto
        r2_teste = 1 - (erro ** 2).sum() / ((teste['preco'] - teste['preco'].mean()) ** 2).sum()

        c = st.columns(6)
        c[0].metric('R² no treino', f'{modelo.rsquared:.3f}')
        c[1].metric('R² ajustado', f'{modelo.rsquared_adj:.3f}')
        c[2].metric('R² no teste (2010)', f'{r2_teste:.3f}')
        c[3].metric('RMSE no teste (US$)', f'{np.sqrt((erro ** 2).mean()):,.0f}')
        c[4].metric('MAE no teste (US$)', f'{erro.abs().mean():,.0f}')
        c[5].metric('Casas: treino / teste', f'{len(treino)} / {len(teste)}')

        intervalos = modelo.conf_int(alpha=alfa)
        coeficientes = pd.DataFrame({'Coeficiente': modelo.params, f'IC {confianca:.0%} inferior': intervalos[0],
                                     f'IC {confianca:.0%} superior': intervalos[1], 'Estatística t': modelo.tvalues,
                                     'p-valor': modelo.pvalues}).drop('Intercept')
        coeficientes.index = [rotulo(i) for i in coeficientes.index]
        st.dataframe(coeficientes.style.format('{:,.2f}', subset=coeficientes.columns[:4]).format(texto_p, subset=['p-valor']), width='stretch')

        esquerda, direita = st.columns(2)
        limites = [teste['preco'].min(), teste['preco'].max()]
        fig = go.Figure([
            go.Scatter(x=previsto, y=teste['preco'], mode='markers', marker=dict(color=AZUL, opacity=0.6), name='Casas vendidas em 2010',
                       hovertemplate='Previsto: %{x:,.0f}<br>Real: %{y:,.0f}<extra></extra>'),
            go.Scatter(x=limites, y=limites, mode='lines', line=dict(color=VERMELHO, dash='dash'), name='Previsão perfeita')])
        fig.update_layout(title='Preço real × previsto (teste)', xaxis_title='Preço previsto (US$)', yaxis_title='Preço real (US$)', height=430)
        esquerda.plotly_chart(fig, width='stretch')
        fig = px.scatter(x=modelo.fittedvalues, y=modelo.resid, opacity=0.4, color_discrete_sequence=[AZUL],
                         labels={'x': 'Preço previsto (US$)', 'y': 'Resíduo (US$)'}, title='Resíduos × valores previstos (treino)')
        fig.add_hline(0, line_color=VERMELHO)
        fig.update_layout(height=430)
        direita.plotly_chart(fig, width='stretch')

        st.subheader('Simulador de preço')
        entradas = {}
        colunas = st.columns(min(len(escolhidas), 4))
        for i, variavel in enumerate(escolhidas):
            minimo_v, maximo_v = float(dados[variavel].min()), float(dados[variavel].max())
            if variavel in ['area_total', 'area_lote']:
                entradas[variavel] = colunas[i % len(colunas)].slider(rotulo(variavel), minimo_v, maximo_v, float(dados[variavel].median()), step=5.0)
            elif variavel == 'banheiros':
                entradas[variavel] = colunas[i % len(colunas)].slider(rotulo(variavel), minimo_v, maximo_v, 2.0, step=0.5)
            else:
                entradas[variavel] = colunas[i % len(colunas)].slider(rotulo(variavel), int(minimo_v), int(maximo_v), int(dados[variavel].median()))
        resultado = modelo.get_prediction(pd.DataFrame([entradas])).summary_frame(alpha=alfa).iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric('Preço previsto (US$)', f'{resultado["mean"]:,.0f}')
        c2.metric(f'IC {confianca:.0%} do preço médio', f'{resultado["mean_ci_lower"]:,.0f} a {resultado["mean_ci_upper"]:,.0f}')
        c3.metric(f'Intervalo de predição {confianca:.0%}', f'{resultado["obs_ci_lower"]:,.0f} a {resultado["obs_ci_upper"]:,.0f}')
