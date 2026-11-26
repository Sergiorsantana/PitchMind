# Como contribuir com o PitchMind

Obrigado pelo interesse! Contribuições de código, dados, avaliação e documentação são bem-vindas.

## Ambiente

```bash
git clone https://github.com/Sergiorsantana/PitchMind.git
cd PitchMind
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"
pytest -q
```

## Fluxo

1. Abra uma *issue* descrevendo o problema ou a ideia (use os modelos).
2. Crie uma branch a partir da `main`: `feat/...`, `fix/...` ou `docs/...`.
3. Escreva testes para o que mudou e rode `pytest -q` e `pitchmind --backend offline eval`.
4. Abra o *pull request* explicando **o que** mudou e **como foi medido**.

## Princípios do projeto

- **Todo número tem lastro.** Métricas vêm do tracking; o LLM/VLM interpreta, mas nunca inventa números.
- **Abster é melhor que errar.** Na dúvida, a identidade fica desconhecida.
- **Resultado reproduzível.** Toda afirmação de melhoria no README vem de um comando que qualquer um roda.
- **Honestidade sobre os dados.** Diga sempre se um resultado vem de dados simulados ou reais.

## Ideias em aberto

- Rodar `pitchmind bench-jersey` no SoccerNet real e publicar os números.
- Testar a leitura de vídeo real (`pitchmind video`) com um modelo de keypoints do campo.
- Novos eventos táticos (por exemplo, linha de impedimento e superioridade numérica por zona).
