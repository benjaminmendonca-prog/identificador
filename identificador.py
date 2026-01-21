import os
import shutil
import customtkinter as ctk 
from tkinter import filedialog, messagebox
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch
import threading
# --- CONFIGURAÇÕES ---
PASTA_BANCO = "./banco_de_imagens"
MODELO_NOME = 'clip-ViT-B-32'

# Configuração do Tema (Dark Mode e Cor Azul)
ctk.set_appearance_mode("Dark")  
ctk.set_default_color_theme("blue")  

class AppModerno(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configurações da Janela
        self.title("Neural Search - Identificador Forense")
        self.geometry("1000x700")

        # Variáveis
        self.caminho_imagem_atual = None
        self.embeddings_banco = None
        self.nomes_arquivos_banco = []
        self.model = None

        # Criar pasta se não existir
        if not os.path.exists(PASTA_BANCO):
            os.makedirs(PASTA_BANCO)

        # --- LAYOUT (GRID) ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Painel Lateral (Esquerda - Controles)
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Neural Detect 👁️", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_carregar = ctk.CTkButton(self.sidebar_frame, text="📂 Carregar Imagem", command=self.carregar_imagem)
        self.btn_carregar.grid(row=1, column=0, padx=20, pady=10)

        self.separador = ctk.CTkLabel(self.sidebar_frame, text="Adicionar ao Banco:", anchor="w")
        self.separador.grid(row=2, column=0, padx=20, pady=(20,0))

        self.entry_nome = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Nome (ex: logo_gangue)")
        self.entry_nome.grid(row=3, column=0, padx=20, pady=5)

        self.btn_salvar = ctk.CTkButton(self.sidebar_frame, text="💾 Salvar e Treinar", fg_color="green", hover_color="darkgreen", command=self.salvar_no_banco, state="disabled")
        self.btn_salvar.grid(row=4, column=0, padx=20, pady=10, sticky="n")

        self.lbl_status = ctk.CTkLabel(self.sidebar_frame, text="Iniciando...", text_color="gray")
        self.lbl_status.grid(row=5, column=0, padx=20, pady=20)

        # 2. Área Principal (Direita - Visualização)
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        # Imagem Preview
        self.img_preview_frame = ctk.CTkFrame(self.main_frame)
        self.img_preview_frame.pack(fill="both", expand=True, pady=(0, 20))
        
        self.lbl_imagem = ctk.CTkLabel(self.img_preview_frame, text="Nenhuma imagem selecionada")
        self.lbl_imagem.pack(fill="both", expand=True, padx=10, pady=10)

        # Caixa de Resultados
        self.lbl_resultado_titulo = ctk.CTkLabel(self.main_frame, text="Análise de Similaridade:", font=ctk.CTkFont(size=16, weight="bold"), anchor="w")
        self.lbl_resultado_titulo.pack(fill="x")
        
        self.txt_resultado = ctk.CTkTextbox(self.main_frame, height=150)
        self.txt_resultado.pack(fill="x")

        # Iniciar carregamento da IA em outra thread para não travar a tela
        threading.Thread(target=self.carregar_modelo, daemon=True).start()

    def carregar_modelo(self):
        self.lbl_status.configure(text="Carregando IA...")
        try:
            self.model = SentenceTransformer(MODELO_NOME)
            self.atualizar_banco_dados()
            self.lbl_status.configure(text="Sistema Online", text_color="#00ff00")
        except Exception as e:
            self.lbl_status.configure(text="Erro na IA", text_color="red")
            print(e)

    def atualizar_banco_dados(self):
        imgs_pil = []
        self.nomes_arquivos_banco = []

        if os.path.exists(PASTA_BANCO):
            arquivos = os.listdir(PASTA_BANCO)
            for arquivo in arquivos:
                if arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        caminho = os.path.join(PASTA_BANCO, arquivo)
                        img = Image.open(caminho)
                        imgs_pil.append(img)
                        self.nomes_arquivos_banco.append(arquivo)
                    except:
                        continue
        
        if imgs_pil:
            self.embeddings_banco = self.model.encode(imgs_pil, convert_to_tensor=True)
            print(f"Banco atualizado: {len(imgs_pil)} imagens")
        else:
            self.embeddings_banco = None

    def carregar_imagem(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return

        self.caminho_imagem_atual = file_path
        
        # Exibir Imagem usando CTkImage (melhor qualidade)
        img_pil = Image.open(file_path)
        # Mantém a proporção
        razao = img_pil.height / img_pil.width
        nova_largura = 400
        nova_altura = int(nova_largura * razao)
        
        my_image = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(nova_largura, nova_altura))
        self.lbl_imagem.configure(image=my_image, text="")
        
        self.btn_salvar.configure(state="normal")
        self.analisar_imagem()

    def analisar_imagem(self):
        if self.model is None:
            return

        if self.embeddings_banco is None:
            self.txt_resultado.delete("0.0", "end")
            self.txt_resultado.insert("0.0", "Banco vazio. Salve esta imagem para iniciar o aprendizado.")
            return

        img = Image.open(self.caminho_imagem_atual)
        embedding_input = self.model.encode(img, convert_to_tensor=True)

        scores = util.cos_sim(embedding_input, self.embeddings_banco)[0]
        top_results = torch.topk(scores, k=min(5, len(scores)))

        self.txt_resultado.delete("0.0", "end")
        
        for score, idx in zip(top_results[0], top_results[1]):
            score_val = score.item() * 100
            nome = self.nomes_arquivos_banco[idx]
            
            # Formatação do texto de resultado
            texto_linha = f"{score_val:.1f}%  ➜  {nome}\n"
            self.txt_resultado.insert("end", texto_linha)

    def salvar_no_banco(self):
        novo_nome = self.entry_nome.get().strip()
        if not novo_nome:
            messagebox.showwarning("Atenção", "Digite um nome para a referência.")
            return

        if not novo_nome.lower().endswith(('.jpg', '.png', '.jpeg')):
            novo_nome += ".jpg"

        destino = os.path.join(PASTA_BANCO, novo_nome)
        
        try:
            shutil.copy(self.caminho_imagem_atual, destino)
            self.entry_nome.delete(0, "end")
            self.atualizar_banco_dados()
            self.analisar_imagem() # Reanalisa para mostrar 100%
            messagebox.showinfo("Sucesso", "Imagem salva e indexada!")
            
        except Exception as e:
            messagebox.showerror("Erro", str(e))

if __name__ == "__main__":
    app = AppModerno()
    app.mainloop()