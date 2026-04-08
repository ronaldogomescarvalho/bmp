# rPPG HRV Monitor — Deploy Web (Railway)

## Estrutura do projeto

```
rppg_web/
├── app.py                  ← servidor Flask + WebSocket
├── face_landmarker.task    ← modelo MediaPipe (copiar do projeto original)
├── templates/
│   └── index.html          ← interface do celular
├── requirements.txt
├── Procfile
├── railway.json
└── .gitignore
```

---

## Passo 1 — Copiar o modelo MediaPipe

Copie o arquivo `face_landmarker.task` do seu projeto original
para a raiz desta pasta (ao lado do `app.py`).

---

## Passo 2 — Criar repositório no GitHub

```bash
cd rppg_web
git init
git add .
git commit -m "primeiro commit"

# Crie um repositório no github.com e depois:
git remote add origin https://github.com/SEU_USUARIO/rppg-web.git
git push -u origin main
```

---

## Passo 3 — Deploy no Railway

1. Acesse https://railway.app e crie uma conta (pode entrar com GitHub)
2. Clique em **New Project** → **Deploy from GitHub repo**
3. Selecione o repositório `rppg-web`
4. O Railway detecta automaticamente o `Procfile` e faz o deploy
5. Aguarde o build (primeira vez leva ~3-5 minutos — instala mediapipe)
6. Quando aparecer ✅, clique no serviço → **Settings** → **Networking**
   → **Generate Domain** para pegar a URL temporária (ex: `rppg-web.up.railway.app`)

---

## Passo 4 — Apontar seu domínio

### No Railway:
1. Settings → Networking → **Custom Domain**
2. Digite seu domínio (ex: `rppg.seusite.com`)
3. O Railway mostra um valor CNAME (ex: `rppg-web.up.railway.app`)

### No seu registrador de domínio (GoDaddy / Registro.br / Cloudflare etc.):
1. Vá em DNS / Gerenciar DNS
2. Adicione um registro **CNAME**:
   - Nome: `rppg` (ou `@` para domínio raiz, ou `www`)
   - Valor: o CNAME que o Railway forneceu
   - TTL: 3600 (ou automático)
3. Aguarde até 24h para propagar (geralmente < 1h)

O Railway provisiona HTTPS automático via Let's Encrypt. ✅

---

## Passo 5 — Testar no celular

1. Abra o domínio no Chrome ou Safari do celular
2. O browser pedirá permissão de câmera → **Permitir**
3. Pressione **Iniciar**, posicione o rosto e aguarde ~15s para o BPM aparecer

---

## Variáveis de ambiente (opcional)

No Railway → Variables, você pode definir:

| Variável     | Descrição                  | Padrão          |
|--------------|----------------------------|-----------------|
| SECRET_KEY   | Chave secreta Flask        | rppg-secret-2024|
| PORT         | Porta (Railway define auto)| 5000            |

---

## Solução de problemas

**Câmera não abre no celular:**
- Verifique se está acessando via HTTPS (obrigatório)
- No Chrome Android: toque no cadeado → Permissões → Câmera → Permitir

**Build falha no Railway:**
- Verifique se o `face_landmarker.task` está commitado no repositório
- O arquivo é grande (~29MB) — o Railway suporta, mas o GitHub tem limite de 100MB

**BPM não aparece:**
- Precisa de pelo menos ~2s de dados com rosto detectado
- Boa iluminação frontal é essencial
- Use câmera traseira para melhor qualidade de imagem

**WebSocket não conecta:**
- O Railway suporta WebSocket nativamente
- Certifique-se que o Procfile usa `eventlet` como worker
