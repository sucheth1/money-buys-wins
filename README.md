# Does Money Buy Wins? MLB payroll and winning, 1985–2016

Quarto report using the Lahman Baseball Database (918 team-seasons).

- `report.qmd`: the report (Python/Jupyter engine)
- `data/lahman_wins_payroll_1985_2016.csv`: wins, payroll and playoff results by team-season

## Render

```bash
pip install -r requirements.txt
quarto render report.qmd
```

This produces `report.html` (self-contained) and `report.md` (renders on GitHub).
