# Projeto SAEB — desigualdade socioeconômica, território e educação (9º ano)

Este projeto transforma microdados do SAEB e do Censo Escolar em uma base analítica por escola,
tabelas descritivas, modelos de regressão e um dashboard Streamlit, para investigar a relação entre
nível socioeconômico (INSE) + região e o desempenho em Matemática e Português no 9º ano do
Ensino Fundamental, segundo os níveis de proficiência do SAEB.

## Como executar

```bash
pip install -r requirements.txt
python pipeline.py --config config.json
streamlit run dashboard.py
```

Coloque os arquivos do SAEB (`saeb.csv`), do resultado por escola (`TS_ESCOLA.csv`) e do Censo
Escolar (`censo_escolar.csv`) em `data/raw/` e ajuste `config.json` se os nomes de coluna do seu
ano de referência forem diferentes.

## Cuidados metodológicos

- O escore usado é `PROFICIENCIA_MT_SAEB` / `PROFICIENCIA_LP_SAEB` (escala SAEB, 0–500), não o
  `PROFICIENCIA_MT` / `PROFICIENCIA_LP` (theta/TRI padronizado). Só a escala SAEB é compatível com
  os níveis de proficiência definidos pelo INEP.
- Os cortes de nível de proficiência (`proficiency_levels` em `config.json`) são validados
  automaticamente a cada execução contra os percentuais oficiais `NIVEL_*_MT9`/`NIVEL_*_LP9` do
  `TS_ESCOLA.csv`; o resultado fica em `results/validacao_niveis_proficiencia.txt` (erro médio
  observado: abaixo de 0,2 ponto percentual).
- **`ID_ESCOLA`/`ID_MUNICIPIO` em `saeb.csv` não usam o mesmo espaço de códigos que
  `CO_ENTIDADE`/`CO_MUNICIPIO` em `censo_escolar.csv`** (interseção vazia nos dois casos, verificado
  diretamente nos dados). Por isso não há cruzamento linha a linha por escola com o Censo Escolar
  neste conjunto de dados — o único código compatível entre os dois arquivos é a UF
  (`ID_UF`/`CO_UF`, os 27 códigos batem). O pipeline usa o Censo Escolar apenas para montar uma
  tabela pequena de código → nome de UF/região, aplicada sobre os códigos que o próprio SAEB já
  traz. Região, estado, localização (urbana/rural) e rede (pública/privada) vêm inteiramente do
  `saeb.csv`. Se você tiver acesso aos microdados oficiais do INEP (onde os códigos de escola
  realmente coincidem entre bases), um cruzamento por escola traria mais precisão.
- Os pesos amostrais são aplicados por disciplina (`PESO_ALUNO_MT` para Matemática,
  `PESO_ALUNO_LP` para Português, `PESO_ALUNO_INSE` para o nível socioeconômico) — não existe um
  peso único genérico nos microdados.
- O pipeline filtra os alunos pela série informada em `target_grade` (9º ano) antes de agregar.
- Os resultados são associações observacionais, não provas de causalidade.
- Para estimativas oficiais, consulte o dicionário do ano do SAEB e aplique os pesos amostrais
  conforme o desenho da pesquisa.

## Saídas

- `data/processed/base_analitica.{csv,parquet}`: uma linha por escola, com proficiência média,
  INSE médio, região/UF/localização/rede e o percentual de alunos em cada nível de proficiência.
- `results/`: tabelas descritivas, modelos de regressão (`modelo_desempenho_math.txt`,
  `modelo_desempenho_portuguese.txt` e as versões com interação por região) e o relatório de
  validação dos níveis de proficiência.
- `figures/`: boxplots e dispersões por INSE, barras por região e distribuição de níveis de
  proficiência por nível socioeconômico.
