# Does Money Buy Wins? Payroll and Winning in MLB, 1985–2016
Sucheth Parigi
2026-11-25

## Thesis

**Money doesn’t buy wins. It buys a better lottery ticket.** Across 32
seasons of Major League Baseball, a team’s payroll explains only a small
share of how often it wins. But high payrolls reliably raise a team’s
odds of reaching October and of winning a championship, and the
exceptions (a 103-win Athletics team on a tiny budget, a 43-win Tigers
team that spent more than that, a Yankees team that spent 2.3x the
average and finished 85–77) are what make the story interesting.

## Data

Team wins and playoff results come from the **Lahman Baseball Database**
(`Teams` table). Payroll is the sum of every player’s salary on a team
in a season (`Salaries` table). Lahman’s salary data covers 1985–2016,
which gives **918 team-seasons** with 16 variables each (14,688 cells).

Because payrolls grew a lot over time, the analysis uses **payroll
relative to the league average that season** (1.0 = average), so a 1985
team and a 2016 team are comparable.

``` python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

m = pd.read_csv("data/lahman_wins_payroll_1985_2016.csv")
m["rel"] = m.salary / m.groupby("yearID").salary.transform("mean")
m["wp"] = m.W / (m.W + m.L)

BLUE, ORANGE, GREY, INK = "#2a6fb0", "#d9622b", "#b8bdc4", "#222222"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#999", "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.titlelocation": "left",
})
m[["yearID", "name", "W", "L", "salary", "rel", "Playoffs", "WSWin"]].head()
```



|     | yearID | name              | W   | L   | salary   | rel      | Playoffs | WSWin |
|-----|--------|-------------------|-----|-----|----------|----------|----------|-------|
| 0   | 1985   | Atlanta Braves    | 66  | 96  | 14807000 | 1.469595 | N        | N     |
| 1   | 1985   | Baltimore Orioles | 83  | 78  | 11560712 | 1.147401 | N        | N     |
| 2   | 1985   | Boston Red Sox    | 81  | 81  | 10897560 | 1.081583 | N        | N     |
| 3   | 1985   | California Angels | 90  | 72  | 14427894 | 1.431969 | N        | N     |
| 4   | 1985   | Chicago White Sox | 85  | 77  | 9846178  | 0.977233 | N        | N     |

</div>

## 1. Payroll helps, but it explains little

``` python
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(m.rel, m.wp, s=22, color=GREY, alpha=.75, edgecolor="white", lw=.4)
b = np.polyfit(m.rel, m.wp, 1)
xs = np.linspace(m.rel.min(), m.rel.max(), 50)
ax.plot(xs, np.polyval(b, xs), color=BLUE, lw=2)

picks = [(2002, "OAK", (.08, -.04)), (2001, "SEA", (-.55, .025)),
         (2003, "DET", (.1, .02)), (1998, "NYA", (.12, .02)),
         (2013, "NYA", (-.1, -.05))]
for y, t, off in picks:
    r = m[(m.yearID == y) & (m.teamID == t)].iloc[0]
    ax.scatter(r.rel, r.wp, s=50, color=ORANGE, zorder=3)
    ax.annotate(f'{r["name"].split()[-1]} {y}\n{r.W}-{r.L}, {r.rel:.1f}x avg payroll',
                (r.rel, r.wp), xytext=(r.rel + off[0], r.wp + off[1]), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color="#888", lw=.7))
ax.axhline(.5, color="#ccc", lw=.8, zorder=0)
ax.axvline(1, color="#ccc", lw=.8, zorder=0)
ax.set_ylim(.25, .78)
ax.set_xlabel("Payroll relative to league average that season (1.0 = average)")
ax.set_ylabel("Win %")
ax.set_title("Money helps, but it explains only 14% of the variation in winning")
r2 = np.corrcoef(m.rel, m.wp)[0, 1] ** 2
ax.text(.99, .03, f"Each +1x average payroll ≈ +{b[0]*162:.0f} wins/162 games · R² = {r2:.2f} · n = {len(m)}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#555")
fig.tight_layout(); plt.show()
```

![Win percentage vs. payroll relative to league average, 918
team-seasons.](report_files/figure-commonmark/cell-3-output-1.png)

The fitted line says that each additional 1.0x of league-average payroll
is worth about 12 wins over a 162-game season, but the cloud of points
is wide. The outliers show how much room there is for surprises:

- **Athletics, 2002:** 103 wins on 0.6x the average payroll.
- **Mariners, 2001:** 116 wins on about average payroll.
- **Tigers, 2003:** 43–119 on 0.7x the average payroll.
- **Yankees, 2013:** 85–77 on 2.3x the average payroll.

## 2. But money changes your odds

``` python
m["q"] = pd.qcut(m.rel, 4, labels=["Lowest\nquarter", "Second", "Third", "Highest\nquarter"])
g = m.groupby("q", observed=True).agg(
    n=("W", "size"),
    po=("Playoffs", lambda s: (s == "Y").mean() * 100),
    ws=("WSWin", lambda s: (s == "Y").sum()))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5))
for a, v, lab, fmt in [(a1, g.po, "Teams reaching the playoffs (%)", "{:.0f}%"),
                       (a2, g.ws, "World Series titles (of 31 played)", "{:.0f}")]:
    a.bar(range(4), v, color=[GREY, GREY, GREY, BLUE], width=.65)
    for i, x in enumerate(v):
        a.text(i, x + v.max() * .02, fmt.format(x), ha="center", fontsize=11, fontweight="bold")
    a.set_xticks(range(4)); a.set_xticklabels(g.index)
    a.set_title(lab, fontsize=11); a.set_ylim(0, v.max() * 1.15)
    a.spines["left"].set_visible(False); a.set_yticks([])
fig.suptitle("Top-spending quarter wins the title 8x as often as the bottom",
             x=.01, ha="left", fontweight="bold", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .94)); plt.show()
g.round(1)
```

![Playoff rate and World Series titles by payroll
quarter.](report_files/figure-commonmark/cell-4-output-1.png)



|                  | n   | po   | ws  |
|------------------|-----|------|-----|
| q                |     |      |     |
| Lowest           | 230 | 11.3 | 2   |
| Second           | 229 | 17.9 | 4   |
| Third            | 229 | 26.6 | 8   |
| Highest          | 230 | 40.9 | 17  |



Teams in the top payroll quarter reached the playoffs 41% of the time
versus 11% for the bottom quarter, and won 17 of the 31 championships
versus 2 for the bottom quarter. Spending simply buys better odds.

## 3. Yet the champion is rarely the biggest spender

``` python
c = m[m.WSWin == "Y"].sort_values("yearID")
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.axhline(1, color="#aaa", lw=1, ls="--")
ax.text(2016.6, 1.0, "league\naverage", fontsize=8.5, va="center", color="#666")
ax.scatter(m.yearID, m.rel, s=10, color="#d5d9de", zorder=1)
ax.scatter(c.yearID, c.rel, s=70, color=[ORANGE if r < 1 else BLUE for r in c.rel],
           zorder=3, edgecolor="white")
for r in c.itertuples():
    if r.rel < .93 or r.rel > 1.9:
        ax.annotate(r.name.split()[-1], (r.yearID, r.rel),
                    xytext=(0, 9 if r.rel > 1 else -14), textcoords="offset points",
                    ha="center", fontsize=8.5)
ax.set_xlabel("Season"); ax.set_ylabel("Payroll relative to league average")
ax.set_xlim(1984, 2018)
ax.set_title(f"{(c.rel > 1).sum()} of 31 champions out-spent the average team; {(c.rel < 1).sum()} did not")
plt.tight_layout(); plt.show()
c[c.rel < 1][["yearID", "name", "rel"]].round(2)
```

![Payroll of every World Series champion (large dots) against all other
teams (grey).](report_files/figure-commonmark/cell-5-output-1.png)



|     | yearID | name               | rel  |
|-----|--------|--------------------|------|
| 10  | 1985   | Kansas City Royals | 0.93 |
| 64  | 1987   | Minnesota Twins    | 0.61 |
| 136 | 1990   | Cincinnati Reds    | 0.84 |
| 168 | 1991   | Minnesota Twins    | 0.99 |
| 468 | 2002   | Anaheim Angels     | 0.91 |
| 509 | 2003   | Florida Marlins    | 0.70 |
| 869 | 2015   | Kansas City Royals | 0.96 |

</div>

Most champions out-spent the average team, but seven did not: the 1985
Royals, 1987 Twins, 1990 Reds, 1991 Twins (at essentially average
payroll), 2002 Angels, 2003 Marlins and 2015 Royals. The title is a
lottery that the rich enter with more tickets, not one they control.

## 4. The link has shifted over time

``` python
cy = m.groupby("yearID").apply(lambda d: d.wp.corr(d.rel), include_groups=False)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(cy.index, cy.values, color=GREY, width=.7)
ax.plot(cy.index, cy.rolling(5, center=True).mean(), color=BLUE, lw=2.5)
ax.axhline(0, color="#888", lw=.8)
ax.set_ylabel("Correlation: payroll vs. win % (within season)")
ax.set_xlabel("Season")
ax.set_title("The link between payroll and wins strengthened after the mid-1990s")
ax.text(1985, .62, f"1985-99 average r = {cy.loc[1985:1999].mean():.2f}\n"
                   f"2000-16 average r = {cy.loc[2000:2016].mean():.2f}", fontsize=10)
ax.text(2016.6, .2, "5-year\nrolling avg", color=BLUE, fontsize=8.5)
plt.tight_layout(); plt.show()
```

![Within-season correlation between relative payroll and win %, by
year.](report_files/figure-commonmark/cell-6-output-1.png)

Payroll mattered more after the mid-1990s: the average within-season
correlation rose from 0.31 in 1985–99 to 0.40 in 2000–16. Individual
years are noisy, so the two-period averages and the rolling line are the
safer evidence.

## Conclusion

Payroll is a real but modest advantage. It explains about 14% of the
variation in win percentage, yet it moves a team from an 11% to a 41%
chance of making the playoffs and from 2 to 17 championships across
payroll quarters. Baseball’s randomness leaves room for the Athletics
and the Tigers.


## Data source

Lahman Baseball Database (Sean Lahman, SABR), `Teams` and `Salaries`
tables, accessed via the `pylahman` Python package.
<https://sabr.org/lahman-database/>
