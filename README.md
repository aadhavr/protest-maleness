# Who Shows Up

Code for "Who Shows Up: exam leaks, protest, and the visible maleness of South Asia's youth politics" (Aadhav Rajesh, 2026). The write-up is in `post/index.qmd` and at [aadhavrajesh.com](https://www.aadhavrajesh.com).

The project asks three questions about competitive-exam leaks in India, 2023 to 2026:

1. Do leaks cause protest, and by whom?
2. Why do exam protests look so male, and how much of that is who attends versus how protests are recorded?
3. Does the grievance spill over into resentment of women online?

## Design

The main estimate is a stacked event study over nine exam leaks and 640 districts (Census 2011 boundaries), weeks -8 to +8 around each exam date:

$$
y_{dt} = \sum_{k \neq -1} \beta_k \, A_{de} \, \mathbf{1}[t - T_e = k] + \alpha_{de} + \gamma_{te} + \varepsilon_{dt}
$$

$y_{dt}$ is the number of exam protests in district $d$ and week $t$, and $A_{de}$ is applicants to exam $e$ per 1,000 residents aged 15-29 (zero outside the exposed state). Because treatment varies across only six states and one national exam, the reported p-values come from randomisation inference: all exam dates are shifted together by every eligible number of weeks (164 shifts).

Main result: non-party exam protests rise after a leak in proportion to exposure ($\hat\beta = 0.00045$, randomisation $p = 0.048$). Party-led protests do not ($\hat\beta = 0.00014$, $p = 0.27$). Robustness checks, deviations from the original plan, and the methods for parts 2 and 3 are in the appendix of the write-up.

## Data

None of the raw data are in this repository. Sources and how to get them:

| Data | Source | Access |
|---|---|---|
| Protest events | [ACLED](https://acleddata.com) | Free account; `collect/c01_acled_api.py`. ACLED's terms do not allow redistribution of event-level data. |
| Exam leaks | GDELT daily files, then official notices and news reports | `collect/c02b_gdelt_bulk.py` finds candidates; each shock was verified by hand. The final list is in the write-up (appendix A1). |
| Applicants | Exam boards and NTA press notes | Collected by hand; template in `templates/applicants.csv`. |
| Population | Census of India 2011 (C-13, PCA) | `collect/c00_census_download.py`, `collect/c03_census.py` |
| District boundaries | [DataMeet](https://github.com/datameet/maps) | `collect/c10_acled_spatial_xwalk.py` |
| Court cases | [Indian High Court Judgments](https://registry.opendata.aws/indian-high-court-judgments/), Dattam Labs (CC BY 4.0) | `collect/c04_courts_s3.py`, no account needed |
| Searches | Google Trends | `collect/c08_trends_pull.py` |
| Comments | YouTube Data API v3 | `collect/c06_youtube_comments.py` |
| Surveys | World Values Survey Wave 7; Pew Research Center India survey (2019-20) | Download from WVS and Pew; place in `data/raw/surveys/` |

Text classification (exam protests, petitioner sex, comment blame) uses the Anthropic API (`collect/c09_classify_llm.py`), with prompts and rules in the script. Hand validation is in `14_validate_labels.py`.

API keys are read from environment variables: `ACLED_USERNAME`, `ACLED_PASSWORD`, `YOUTUBE_API_KEY`, `ANTHROPIC_API_KEY`.

## Reproducing

Python 3.11.

```
pip install -r requirements.txt
./run_collect.sh     # once; several steps need keys or files downloaded by hand
./run_analysis.sh    # everything reported in the write-up
```

Outputs go to `data/output/`. To check that the pipeline runs without the real data:

```
python3 tests/make_fake_data.py
PM_DATA_ROOT=tests/fake_data ./run_analysis.sh
```

The fake data have a planted effect, so the estimates from this run mean nothing.

## Layout

```
collect/        data collection (c00-c10)
01-05c          Part 1: shocks, ACLED panel, exposure, stacked panel, event study, randomisation
07-09           state-level panel: courts and Google Trends
16, 18          Part 2: courts vs street, how exam protests are reported
12, 22, 23      Part 3: comments, Trends detectable effects, surveys
17, 24          robustness, effect sizes
20              Bangladesh and Nepal comparison
26, 27          figures for the write-up
config.py       settings (window, event date, terms)
lib.py          estimators (fixed-effects demeaning, clustered errors)
templates/      blank versions of the hand-built input files
post/           the write-up
```

Scripts 06, 10, 11, 13, 15, 19 and 21 are exploratory and not reported.

## Citation

Rajesh, Aadhav. 2026. "Who Shows Up: Exam Leaks, Protest, and the Visible Maleness of South Asia's Youth Politics." https://www.aadhavrajesh.com

Please also cite ACLED, the World Values Survey, Pew Research Center and the court data as their terms require.
