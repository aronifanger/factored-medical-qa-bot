import streamlit as st
import requests

# --- Configuração da Página ---
st.set_page_config(
    page_title="Medical QA Bot",
    page_icon="🤖",
    layout="centered"
)

# --- Título e Descrição ---
st.title("🤖 Medical QA Bot")
st.markdown("""
Bem-vindo ao Medical QA Bot! Este bot foi treinado para responder a perguntas sobre doenças médicas.
**Como usar:**
1.  Digite sua pergunta na caixa de texto abaixo.
2.  Clique no botão "Perguntar".
3.  A resposta e o contexto usado para gerá-la serão exibidos.
""")

# --- Endereço da API ---
API_URL = "http://127.0.0.1:8000/ask"

# --- Interação com o Usuário ---
question = st.text_input("Qual é a sua pergunta médica?", "")

if st.button("Perguntar"):
    if question:
        try:
            # --- Chamada à API ---
            with st.spinner("Buscando a resposta..."):
                response = requests.post(API_URL, json={"question": question})
                response.raise_for_status()  # Lança um erro para códigos de status HTTP ruins (4xx ou 5xx)

                data = response.json()

                # --- Exibir a Resposta ---
                st.subheader("Resposta:")
                st.success(data['answer'])

                with st.expander("Ver Contexto Utilizado"):
                    st.info(data['context'])
                
                st.write(f"**Score de Confiança:** {data['score']:.4f}")

        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao conectar com a API: {e}")
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado: {e}")
    else:
        st.warning("Por favor, digite uma pergunta.")

# --- Rodapé ---
st.markdown("---")
st.markdown("Desenvolvido como parte do desafio de codificação.") 