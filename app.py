# ============================================================
# PREVISOR DE DEMANDA SEMANAL
# Aplicativo desenvolvido para a disciplina de Administração da Produção
# Tema: Previsão de demanda para apoio ao planejamento da produção
# ============================================================
# Este código foi gerado com apoio de IA generativa (metodologia "vibe coding").
# Cada bloco abaixo tem um comentário explicando o que ele faz, em linguagem simples,
# para que qualquer pessoa sem experiência em programação consiga acompanhar.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ------------------------------------------------------------
# BLOCO 1 - FUNÇÕES DE PREVISÃO
# Cada função abaixo recebe a série histórica (lista de números)
# e devolve uma lista com as previsões futuras.
# ------------------------------------------------------------

def previsao_ingenua(serie, n_futuras):
    """Método ingênuo: a previsão futura repete o último valor conhecido."""
    ultimo_valor = serie[-1]
    return [ultimo_valor] * n_futuras


def previsao_media_movel_simples(serie, n_futuras, janela=3):
    """Média móvel simples: previsão = média das últimas 'janela' semanas.
    Conforme a previsão avança, o próprio valor previsto entra na média seguinte."""
    serie_estendida = list(serie)
    previsoes = []
    for _ in range(n_futuras):
        ultimos = serie_estendida[-janela:]
        media = sum(ultimos) / len(ultimos)
        previsoes.append(media)
        serie_estendida.append(media)
    return previsoes


def previsao_media_movel_ponderada(serie, n_futuras, janela=3):
    """Média móvel ponderada: semanas mais recentes pesam mais.
    Para janela=3, os pesos usados são 0.2 / 0.3 / 0.5 (do mais antigo para o mais recente)."""
    pesos = np.array([0.2, 0.3, 0.5]) if janela == 3 else np.linspace(1, 2, janela)
    pesos = pesos / pesos.sum()  # garante que os pesos somem 1

    serie_estendida = list(serie)
    previsoes = []
    for _ in range(n_futuras):
        ultimos = np.array(serie_estendida[-janela:])
        media_ponderada = float(np.dot(ultimos, pesos))
        previsoes.append(media_ponderada)
        serie_estendida.append(media_ponderada)
    return previsoes


def previsao_suavizacao_exponencial(serie, n_futuras, alfa=0.4):
    """Suavização exponencial simples: combina a última demanda real com a
    previsão anterior. Fórmula: F(t+1) = alfa * D(t) + (1 - alfa) * F(t)."""
    F = serie[0]  # primeira previsão começa igual ao primeiro valor da série
    for valor_real in serie[1:]:
        F = alfa * valor_real + (1 - alfa) * F
    # F agora é a previsão para a próxima semana; repetimos o mesmo valor
    # suavizado para as semanas futuras seguintes (abordagem simples e didática)
    return [F] * n_futuras


def previsao_regressao_linear(serie, n_futuras):
    """Regressão linear simples: ajusta uma reta (tendência) sobre a série
    histórica e projeta essa reta para as semanas futuras."""
    x = np.arange(len(serie))
    y = np.array(serie)
    # np.polyfit encontra os coeficientes 'a' (inclinação) e 'b' (intercepto) da reta y = a*x + b
    a, b = np.polyfit(x, y, 1)
    x_futuro = np.arange(len(serie), len(serie) + n_futuras)
    previsoes = a * x_futuro + b
    return previsoes.tolist(), a  # devolve também a inclinação, útil para saber a tendência


# ------------------------------------------------------------
# BLOCO 2 - CÁLCULO DE ERRO (para comparar métodos)
# Estratégia: para cada método, simulamos "prever a semana seguinte"
# usando apenas os dados anteriores a ela (walk-forward), e comparamos
# com o valor real. O erro médio absoluto (MAE) resume o desempenho.
# ------------------------------------------------------------

def calcular_mae(serie, funcao_previsao, **kwargs):
    """Calcula o Erro Médio Absoluto (MAE) de um método, testando-o
    contra o histórico já conhecido (validação retroativa)."""
    erros = []
    minimo_pontos = 4  # precisa de um mínimo de histórico para começar a testar
    if len(serie) <= minimo_pontos:
        return None  # dados insuficientes para calcular erro com confiança

    for corte in range(minimo_pontos, len(serie)):
        historico_parcial = serie[:corte]
        valor_real = serie[corte]
        try:
            resultado = funcao_previsao(historico_parcial, 1, **kwargs)
            previsto = resultado[0][0] if isinstance(resultado, tuple) else resultado[0]
        except Exception:
            continue
        erros.append(abs(valor_real - previsto))

    if not erros:
        return None
    return float(np.mean(erros))


# ------------------------------------------------------------
# BLOCO 3 - INTERPRETAÇÃO GERENCIAL
# Transforma números em uma recomendação em linguagem de negócio.
# ------------------------------------------------------------

def gerar_recomendacao_gerencial(serie, previsoes, inclinacao_tendencia):
    media_historica = np.mean(serie)
    media_prevista = np.mean(previsoes)
    variacao_percentual = ((media_prevista - media_historica) / media_historica) * 100

    desvio_padrao = np.std(serie)
    coef_variacao = (desvio_padrao / media_historica) * 100 if media_historica else 0

    mensagens = []

    if coef_variacao > 25:
        mensagens.append(
            "⚠️ A demanda histórica apresenta **alta variação** (irregular). "
            "Recomenda-se investigar fatores externos — sazonalidade, promoções, eventos — "
            "antes de basear decisões apenas nesta previsão."
        )

    if variacao_percentual > 10:
        mensagens.append(
            f"📈 A demanda apresenta **tendência de crescimento** "
            f"(previsão média {variacao_percentual:.1f}% acima da média histórica). "
            "Recomenda-se verificar se a capacidade produtiva atual será suficiente "
            "para atender as próximas semanas, evitando falta de produto."
        )
    elif variacao_percentual < -10:
        mensagens.append(
            f"📉 A demanda apresenta **tendência de queda** "
            f"(previsão média {abs(variacao_percentual):.1f}% abaixo da média histórica). "
            "Recomenda-se cautela na produção para evitar excesso de estoque "
            "e aumento dos custos de armazenagem."
        )
    else:
        mensagens.append(
            "➡️ A demanda apresenta comportamento **relativamente estável**. "
            "A empresa pode usar esta previsão como referência para manter "
            "o planejamento de produção atual."
        )

    mensagens.append(
        "\n*Importante: previsão de demanda é uma estimativa baseada em dados passados, "
        "não uma certeza. O julgamento gerencial deve sempre complementar o número.*"
    )

    return "\n\n".join(mensagens)


# ------------------------------------------------------------
# BLOCO 4 - INTERFACE (o que aparece na tela)
# ------------------------------------------------------------

st.set_page_config(page_title="Previsor de Demanda Semanal", layout="centered")

st.title("📦 Previsor de Demanda Semanal")
st.caption(
    "Aplicativo de apoio ao planejamento da produção. "
    "Informe o histórico de demanda semanal e obtenha previsões para as próximas semanas."
)

# --- Seção 1: entrada de dados ---
st.header("1. Dados do produto")

nome_produto = st.text_input("Nome do produto", value="Produto A")

st.write(
    "Informe as demandas semanais separadas por vírgula (mínimo 8 semanas). "
    "Exemplo: 120, 125, 130, 128, 135, 140, 145, 150"
)
entrada_texto = st.text_area(
    "Demandas históricas (semana 1 → última semana)",
    value="120, 125, 130, 128, 135, 140, 145, 150, 148, 155, 160, 165",
    height=80,
)

col1, col2 = st.columns(2)
with col1:
    n_futuras = st.slider("Quantas semanas futuras prever?", min_value=1, max_value=8, value=4)
with col2:
    metodos_selecionados = st.multiselect(
        "Métodos de previsão",
        ["Ingênuo", "Média móvel simples", "Média móvel ponderada",
         "Suavização exponencial", "Regressão linear"],
        default=["Média móvel simples", "Suavização exponencial", "Regressão linear"],
    )

# --- Configurações avançadas: parâmetros que o PDF do professor menciona
# como ajustáveis (alfa da suavização exponencial e janela da média móvel) ---
with st.expander("⚙️ Configurações avançadas (opcional)"):
    col3, col4 = st.columns(2)
    with col3:
        janela_media_movel = st.slider(
            "Janela da média móvel (quantas semanas usar na média)",
            min_value=2, max_value=6, value=3,
            help="Janelas menores reagem mais rápido a mudanças recentes. "
                 "Janelas maiores suavizam mais, mas atrasam a resposta."
        )
    with col4:
        alfa_suavizacao = st.slider(
            "Alfa da suavização exponencial",
            min_value=0.1, max_value=0.9, value=0.4, step=0.1,
            help="Alfa alto = a previsão reage rápido a mudanças recentes. "
                 "Alfa baixo = a previsão fica mais estável, reage devagar."
        )

# --- Explicação rápida de cada método, para consulta durante a apresentação ---
with st.expander("📘 O que cada método faz (para consulta rápida)"):
    st.markdown("""
| Método | Quando usar | Fórmula resumida |
|---|---|---|
| **Ingênuo** | Referência básica de comparação | Próxima semana = última semana conhecida |
| **Média móvel simples** | Demanda relativamente estável | Média das últimas N semanas |
| **Média móvel ponderada** | Quando dados recentes importam mais | Média com pesos maiores nas semanas recentes |
| **Suavização exponencial** | Demanda com variação moderada | F(t+1) = alfa × D(t) + (1-alfa) × F(t) |
| **Regressão linear** | Demanda com tendência clara de subida/queda | Ajusta uma reta sobre o histórico |
""")

# --- Validação da entrada ---
def validar_e_converter(texto):
    """Converte o texto digitado em uma lista de números, com validação de erros comuns."""
    try:
        valores_brutos = [v.strip() for v in texto.split(",") if v.strip() != ""]
        valores = [float(v) for v in valores_brutos]
    except ValueError:
        return None, "Encontrei um valor que não é um número. Verifique se digitou apenas números separados por vírgula."

    if len(valores) < 8:
        return None, f"Você informou {len(valores)} semanas. São necessárias no mínimo 8 semanas de histórico."

    if any(v < 0 for v in valores):
        return None, "Encontrei valores negativos. Demanda não pode ser negativa — revise os dados."

    return valores, None


if st.button("Calcular previsão", type="primary"):

    serie, erro_msg = validar_e_converter(entrada_texto)

    if erro_msg:
        st.error(erro_msg)
    elif not metodos_selecionados:
        st.warning("Selecione ao menos um método de previsão.")
    else:
        # --- Seção 2: tabela histórica ---
        st.header("2. Histórico informado")
        df_historico = pd.DataFrame({
            "Semana": list(range(1, len(serie) + 1)),
            "Demanda": serie
        })
        st.dataframe(df_historico, use_container_width=True, hide_index=True)

        # --- Cálculo das previsões para cada método selecionado ---
        resultados = {}   # nome_metodo -> lista de previsões
        erros_mae = {}     # nome_metodo -> erro médio absoluto
        inclinacao = None

        if "Ingênuo" in metodos_selecionados:
            resultados["Ingênuo"] = previsao_ingenua(serie, n_futuras)
            erros_mae["Ingênuo"] = calcular_mae(serie, previsao_ingenua)

        if "Média móvel simples" in metodos_selecionados:
            resultados["Média móvel simples"] = previsao_media_movel_simples(serie, n_futuras, janela=janela_media_movel)
            erros_mae["Média móvel simples"] = calcular_mae(serie, previsao_media_movel_simples, janela=janela_media_movel)

        if "Média móvel ponderada" in metodos_selecionados:
            resultados["Média móvel ponderada"] = previsao_media_movel_ponderada(serie, n_futuras, janela=janela_media_movel)
            erros_mae["Média móvel ponderada"] = calcular_mae(serie, previsao_media_movel_ponderada, janela=janela_media_movel)

        if "Suavização exponencial" in metodos_selecionados:
            resultados["Suavização exponencial"] = previsao_suavizacao_exponencial(serie, n_futuras, alfa=alfa_suavizacao)
            erros_mae["Suavização exponencial"] = calcular_mae(serie, previsao_suavizacao_exponencial, alfa=alfa_suavizacao)

        if "Regressão linear" in metodos_selecionados:
            prev_reg, inclinacao = previsao_regressao_linear(serie, n_futuras)
            resultados["Regressão linear"] = prev_reg
            erros_mae["Regressão linear"] = calcular_mae(serie, previsao_regressao_linear)

        # --- Seção 3: tabela de previsões ---
        st.header("3. Previsão para as próximas semanas")
        semanas_futuras = list(range(len(serie) + 1, len(serie) + n_futuras + 1))
        df_previsao = pd.DataFrame({"Semana": semanas_futuras})
        for nome_metodo, valores in resultados.items():
            df_previsao[nome_metodo] = [round(v, 1) for v in valores]
        st.dataframe(df_previsao, use_container_width=True, hide_index=True)

        st.download_button(
            label="⬇️ Baixar previsão em CSV",
            data=df_previsao.to_csv(index=False).encode("utf-8"),
            file_name=f"previsao_{nome_produto.replace(' ', '_')}.csv",
            mime="text/csv",
        )

        # --- Seção 4: gráfico ---
        st.header("4. Gráfico: histórico x previsão")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_historico["Semana"], y=df_historico["Demanda"],
            mode="lines+markers", name="Histórico", line=dict(color="#1f77b4")
        ))
        for nome_metodo, valores in resultados.items():
            fig.add_trace(go.Scatter(
                x=semanas_futuras, y=valores,
                mode="lines+markers", name=nome_metodo, line=dict(dash="dash")
            ))
        fig.update_layout(
            title=f"Demanda de {nome_produto}: histórico e previsão",
            xaxis_title="Semana", yaxis_title="Demanda",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Seção 5: comparação de erro entre métodos ---
        st.header("5. Comparação entre métodos (erro histórico)")
        metodos_com_erro = {k: v for k, v in erros_mae.items() if v is not None}

        if metodos_com_erro:
            df_erros = pd.DataFrame({
                "Método": list(metodos_com_erro.keys()),
                "Erro médio absoluto (MAE)": [round(v, 2) for v in metodos_com_erro.values()]
            }).sort_values("Erro médio absoluto (MAE)")
            st.dataframe(df_erros, use_container_width=True, hide_index=True)

            melhor_metodo = df_erros.iloc[0]["Método"]
            st.info(
                f"📊 Com base no histórico, **{melhor_metodo}** teve o menor erro médio. "
                "Isso **não garante** que será o melhor método para as próximas semanas — "
                "é apenas uma referência baseada no passado."
            )
        else:
            st.warning(
                "Não há dados suficientes para calcular o erro histórico dos métodos "
                "(são necessárias pelo menos 5 semanas)."
            )

        # --- Seção 6: recomendação gerencial ---
        st.header("6. Recomendação gerencial")
        metodo_para_recomendacao = list(resultados.keys())[0]
        previsoes_para_recomendacao = resultados[metodo_para_recomendacao]

        recomendacao = gerar_recomendacao_gerencial(
            serie, previsoes_para_recomendacao, inclinacao
        )
        st.markdown(recomendacao)

else:
    st.info("Preencha os dados acima e clique em **Calcular previsão** para ver os resultados.")

st.markdown("---")
st.caption(
    "Previsor de Demanda Semanal — desenvolvido com apoio de IA generativa "
    "para a disciplina de Administração da Produção."
)
