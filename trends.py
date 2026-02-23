import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from collections import Counter
import ast

df = pd.read_csv("2026_FAB_play_by_play.csv", low_memory=False)
dfDPassing = pd.read_csv("defensivePassingData.csv")

df["is_cover2"] = (df["CoverageType"].dropna() == "Cover 2").astype(int)
df["is_cover4"] = (df["CoverageType"].dropna() == "Cover 4").astype(int)
df["is_cover6"] = (df["CoverageType"].dropna() == "Cover 6").astype(int)
df["is_cover2man"] = (df["CoverageType"].dropna() == "Man Cover 2").astype(int)

df["DefTeam"] = df["DefTeam"].str.strip()
df["OffTeam"] = df["OffTeam"].str.strip()
df["HomeTeam"] = df["HomeTeam"].str.strip()
df["AwayTeam"] = df["AwayTeam"].str.strip()
dfDPassing["Team"] = dfDPassing["Team"].str.strip()

df = df[df["EventType"] == "pass"]

two_high = df[df["CoverageType"].isin([
    "Cover 2",
    "Cover 4",
    "Cover 6",
    "Cover 2 Man"
])]

not_two_high = df[~df["CoverageType"].isin([
    "Cover 2",
    "Cover 4",
    "Cover 6",
    "Cover 2 Man"
])]

#################################################
### Field Side, Down Distribution for Cover 2 ###
#################################################

# fieldside_counts = twoHigh["FieldSide"].value_counts()
# fieldside_pct = twoHigh["FieldSide"].value_counts(normalize=True) * 100
# summary = (
#     twoHigh["FieldSide"]
#     .value_counts()
#     .to_frame(name="count")
#     .assign(percentage=lambda x: x["count"] / x["count"].sum() * 100)
# )
# print(summary)
# summary = (
#     twoHigh["Down"]
#     .value_counts()
#     .to_frame(name="count")
#     .assign(percentage=lambda x: x["count"] / x["count"].sum() * 100)
# )
# print(summary)

# bins = [0, 2, 5, 8, 12, 20, 100]
# labels = ["0–2", "3–5", "6–8", "9–12", "13–20", "20+"]

# df["ToGo_bin"] = pd.cut(df["ToGo"], bins=bins, labels=labels, right=True)

# twoHighToGo = (
#     twoHigh.groupby("ToGo_bin", observed=True)["is_cover2"]
#     .mean()
#     .reset_index(name="twoHighRate")
# )

# twoHighToGo["twoHighRate"] *= 100

# coverage_rates = (
#     df["CoverageType"]
#     .value_counts(normalize=True)
#     .rename("rate")
#     .reset_index()
#     .rename(columns={"index": "CoverageType"})
# )

# coverage_rates["rate"] *= 100

# print(coverage_rates)

# coverage_by_fieldside = (
#     df
#     .loc[df["CoverageType"].notna()]  # optional but recommended
#     .groupby("FieldSide")["CoverageType"]
#     .value_counts(normalize=True)
#     .rename("rate")
#     .reset_index()
# )

# coverage_by_fieldside["rate"] *= 100
# print(coverage_by_fieldside)


##########################################
### Personell impacts on Cover 2 Usage ###
##########################################

# df = df.copy()

# df["personnel"] = (
#     df["RB"].astype(str) + " RB, " +
#     df["TE"].astype(str) + " TE, " +
#     df["WR"].astype(str) + " WR"
# )

# personnel_counts = df["personnel"].value_counts()
# valid_personnel = personnel_counts[personnel_counts > 50].index

# df_filtered = df[df["personnel"].isin(valid_personnel)]

# df_filtered["is_cover2"] = (df_filtered["CoverageType"] == "Cover 2").astype(int)

# personnel_coverage_rates = (
#     df_filtered
#         .groupby(["personnel", "CoverageType"])
#         .size()
#         .reset_index(name="plays")
# )

# personnel_coverage_rates["coverage_rate"] = (
#     personnel_coverage_rates["plays"]
#     / personnel_coverage_rates.groupby("personnel")["plays"].transform("sum")
# ) * 100


# personnel_cover2 = (
#     df_filtered
#         .groupby("personnel")["is_cover2"]
#         .mean()
#         .reset_index(name="cover2_rate")
# )

# personnel_cover2["cover2_rate"] *= 100
# personnel_cover2 = personnel_cover2.sort_values("cover2_rate", ascending=False)
# personnel_cover2 = personnel_cover2.merge(
#     personnel_counts, left_on="personnel", right_index=True
# )

#############################################
#### Field Position Heatmap Cover 2 Usage ###
#############################################

# df = df.loc[df["EventType"] == "pass"].copy()

# df["StartYard_100"] = np.where(
#     df["FieldSide"] == "Own",
#     100 - df["StartYard"],
#     df["StartYard"]
# )

# bins = np.arange(0, 101, 10)
# labels = [f"{i}–{i+9}" for i in range(0, 100, 10)]

# df["StartYard_bin"] = pd.cut(
#     df["StartYard_100"],
#     bins=bins,
#     labels=labels,
#     right=False,
#     include_lowest=True
# )

# two_high = ["Cover 2", "Cover 4", "Cover 6", "Cover 2 Man"]

# df["isTwoHigh"] = df["CoverageType"].isin(two_high).astype(int)

# twoHighFieldpos = (
#     df.groupby("StartYard_100", observed=True)["isTwoHigh"]
#     .mean()
#     .reset_index(name="twoHigh_rate")
# )

# twoHighFieldpos["twoHigh_rate"] *= 100

# print(df["isTwoHigh"].head(10))
# print(df["CoverageType"].head(10))

# yard_index = np.arange(0, 101)

# twoHigh = (
#     twoHighFieldpos
#     .set_index("StartYard_100")
#     .reindex(yard_index)["twoHigh_rate"]
# )

# heatmap_data = twoHigh.values.reshape(1, -1)

# plt.figure(figsize=(18, 3))

# plt.imshow(
#     heatmap_data,
#     aspect="auto",
#     interpolation="nearest"
# )

# plt.colorbar(label="Two-High Coverage Usage (%)")

# plt.gca().invert_xaxis()

# plt.yticks([])

# plt.xticks(
#     ticks=np.arange(0, 101, 10),
#     labels=[str(i) for i in range(0, 101, 10)]
# )

# plt.xlabel("Yards from Opponent's Goal Line")
# plt.title("Two-High Coverage Usage by Field Position")

# plt.show()

#################################################
### Team rates in Cover 2 Defense Usage #########
#################################################

# cover2_by_team = (
#     df.groupby("DefTeam")["is_cover2"]
#     .mean()
#     .reset_index(name="cover2_rate")
# )
# cover2_by_team["cover2_rate"] *= 100

# cover2_by_team = (
#     df.groupby("DefTeam")
#       .agg(
#           cover2_rate=("is_cover2", "mean"),
#           plays=("is_cover2", "size")
#       )
#       .reset_index()
# )

# cover2_by_team["cover2_rate"] *= 100

# print(cover2_by_team.sort_values("cover2_rate", ascending=False))

# import numpy as np

# sizes = np.exp(cover2_by_team["cover2_rate"] / 6)
# sizes = sizes * 40   # scale to reasonable pixel area


# plt.scatter(
#     cover2_by_team["cover2_rate"],
#     cover2_by_team["plays"],
#     s=sizes,
#     alpha=0.7
# )

# # Add team labels
# for _, row in cover2_by_team.iterrows():
#     plt.text(
#         row["cover2_rate"],
#         row["plays"] + 5,
#         row["DefTeam"],
#         fontsize=9,
#         ha="center",
#         va="center"
#     )

# plt.xlabel("Cover 2 Usage Rate (%)")
# plt.ylabel("Defensive Plays")
# plt.title("Cover 2 Usage by Defensive Team")

# plt.grid(True, linestyle="--", alpha=0.4)
# plt.show()

#########################################################
### EPA by Team effectiveness in facing 2 High #########
#########################################################

# two_high = ["Cover 2", "Cover 4", "Cover 6", "Cover 2 Man"]

# df["isTwoHigh"] = df["CoverageType"].dropna().isin(two_high).astype(int)

# df_two_high = df.loc[
#     df["CoverageType"].isin(two_high)
# ].copy()

# two_high_epa_by_team = (
#     df_two_high
#     .groupby("DefTeam")
#     .agg(
#         avg_epa_allowed=("EPA", "mean"),
#         two_high_plays=("EPA", "size")
#     )
#     .reset_index()
#  )

# corr = two_high_epa_by_team["two_high_plays"].corr(
#     two_high_epa_by_team["avg_epa_allowed"]
# )

# r_squared = corr ** 2

# x = two_high_epa_by_team["two_high_plays"]
# y = two_high_epa_by_team["avg_epa_allowed"]

# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, alpha=0.7)

# for _, row in two_high_epa_by_team.iterrows():
#     plt.annotate(
#         row["DefTeam"],
#         (row["two_high_plays"], row["avg_epa_allowed"]),
#         xytext=(0, 6), 
#         textcoords="offset points",
#         ha="center",
#         fontsize=7,
#         alpha=0.8
#     )

# m, b = np.polyfit(x, y, 1)
# plt.plot(x, m*x + b)

# plt.axhline(0, linestyle="--", alpha=0.15, color="grey")

# plt.xlabel("Number of Two-High Safety Plays")
# plt.ylabel("Average EPA Allowed")
# plt.title(
#     "Two-High Coverage Usage vs Effectiveness\n"
#     f"Correlation = {corr:.2f}"
# )

# plt.grid(True, linestyle="--", alpha=0.1)
# plt.show()

# mean_epa = df_two_high["EPA"].mean()
# print(f"Mean EPA allowed on Two-High plays: {mean_epa:.3f}")

# x = sm.add_constant(x)
# model = sm.OLS(y, x).fit()

# print(model.summary())

# residuals = model.resid
# fitted = model.fittedvalues

# plt.figure(figsize=(8, 6))
# plt.scatter(fitted, residuals, alpha=0.7)
# plt.axhline(0, linestyle="--")

# plt.xlabel("Fitted Values")
# plt.ylabel("Residuals")
# plt.title("Residuals vs Fitted Values")
# plt.grid(alpha=0.3)

# plt.show()

# sm.qqplot(residuals, line="45", fit=True)
# plt.title("QQ Plot of Residuals")
# plt.show()

######################################################
### Coverage useage by team #########
######################################################

# df["is_cover2"] = (df["CoverageType"].dropna() == "Cover 2").astype(int)
# df["is_cover4"] = (df["CoverageType"].dropna() == "Cover 4").astype(int)
# df["is_cover6"] = (df["CoverageType"].dropna() == "Cover 6").astype(int)
# df["is_cover2man"] = (df["CoverageType"].dropna() == "Man Cover 2").astype(int)

# two_high_coverages = [
#     "Cover 2",
#     "Cover 4",
#     "Cover 6",
#     "Cover 2 Man"
# ]

# df["is_two_high"] = df["CoverageType"].dropna().isin(two_high_coverages).astype(int)

# two_high_by_team = (
#     df.groupby("DefTeam")["is_two_high"]
#     .mean()
#     .reset_index(name="two_high_rate")
# )
# two_high_by_team["two_high_rate"] *= 100

# two_high_by_team = (
#     df.groupby("DefTeam")
#       .agg(
#           two_high_rate=("is_two_high", "mean"),
#           plays=("is_two_high", "size")
#       )
#       .reset_index()
# )

# # two_high_by_team["two_high_rate"] *= 100

# merged_df = pd.merge(
#     two_high_by_team,
#     dfDPassing,
#     left_on="DefTeam",
#     right_on="Team",
#     how="inner"
# )

# print(merged_df[["DefTeam", "two_high_rate", "INT"]].head())

# corr = merged_df["two_high_rate"].corr(merged_df["INT"])
# print(f"Correlation between two-high rate and interceptions: {corr:.3f}")

# X = sm.add_constant(merged_df["two_high_rate"])
# y = merged_df["INT"]

# model = sm.OLS(y, X).fit()
# print(model.summary())


# print(two_high_by_team.sort_values("two_high_rate", ascending=False))

# sizes = np.exp(two_high_by_team["two_high_rate"] / 6)
# sizes = sizes * 1.5   # scale to reasonable pixel area

# plt.scatter(
#     two_high_by_team["two_high_rate"],
#     two_high_by_team["plays"],
#     s=sizes,
#     alpha=0.7,
#     color = "skyblue"
# )

# # Add team labels
# for _, row in two_high_by_team.iterrows():
#     plt.text(
#         row["two_high_rate"],
#         row["plays"] + 5,
#         row["DefTeam"],
#         fontsize=9,
#         ha="center",
#         va="center"
#     )

# plt.xlabel("Two High Safety Usage Rate (%)")
# plt.ylabel("Defensive Plays")
# plt.title("Two High Safety Usage by Defensive Team")

# plt.grid(True, linestyle="--", alpha=0.4)
# plt.show()

######################################################
### Opp Tendencies vs. Coverage Type #########
######################################################

# percent_Pass = (
#     df.groupby('OffTeam')['EventType']
#       .apply(lambda x: (x == 'pass').mean() * 100)
# )
# percent_Pass.sort_values(ascending=False).plot(kind='bar')
# # plt.ylabel('Percentage (%)')
# # plt.title('Pass Play % by Offensive Team')
# # plt.show()

# cover_types = ['Cover 2', 'Cover 4', 'Cover 6', 'Man Cover 2']

# percent_C246 = (
#     df.groupby('OffTeam')['CoverageType']
#       .apply(lambda x: x.dropna().isin(cover_types).mean() * 100)
# )

# # percent_C246.sort_values(ascending=False).plot(kind='bar')
# # plt.ylabel('Percentage (%)')
# # plt.title('Cover 2/4/6 % by Offensive Team')
# # plt.show()


# scatter_df = pd.DataFrame({
#     'PassPct': percent_Pass,
#     'Cover246Pct': percent_C246
# }) .dropna()


# x = scatter_df["PassPct"]
# y = scatter_df["Cover246Pct"]

# slope, intercept = np.polyfit(x, y, 1)

# x_line = np.linspace(x.min(), x.max(), 100)
# y_line = slope * x_line + intercept

# plt.figure(figsize=(8, 6))

# plt.scatter(x, y)

# plt.plot(
#     x_line,
#     y_line,
#     label=f"Best Fit: y = {slope:.2f}x + {intercept:.2f}",
#     alpha=0.6,
#     color='grey'
# )

# # plt.figure(figsize=(8, 6))
# # plt.scatter(scatter_df['PassPct'], scatter_df['Cover246Pct'])

# for team, row in scatter_df.iterrows():
#     plt.text(row['PassPct'] + 0.3,
#              row['Cover246Pct'] + 0.3,
#              team,
#              fontsize=9)

# plt.xlabel('Pass Play Percentage (%)')
# plt.ylabel('Two High Defense Faced Percentage (%)')
# plt.title('Coverage Faced vs Pass Rate by Offensive Team')
# # plt.show()

# x = scatter_df["PassPct"]
# y = scatter_df["Cover246Pct"]

# x = sm.add_constant(x)  # adds intercept
# model = sm.OLS(y, x).fit()

# print(model.summary())

# residuals = model.resid
# fitted = model.fittedvalues

# # plt.figure(figsize=(8, 6))
# # plt.scatter(fitted, residuals, alpha=0.7)
# # plt.axhline(0, linestyle="--")

# # plt.xlabel("Fitted Values")
# # plt.ylabel("Residuals")
# # plt.title("Residuals vs Fitted Values")
# # plt.grid(alpha=0.3)

# # plt.show()

# sm.qqplot(residuals, line="45", fit=True)
# plt.title("QQ Plot of Residuals")
# plt.show()

######################################################
### Team Offensive and Defensive Correlations ########
######################################################

# two_high_list = [
#     "Cover 2",
#     "Cover 4",
#     "Cover 6",
#     "Cover 2 Man"
# ]

# two_high = df.loc[
#     df["CoverageType"].isin(two_high_list)
# ].copy()

# def_epa_two_high = (
#     two_high
#     .groupby("DefTeam")
#     .agg(def_avg_epa_allowed=("EPA", "mean"),
#          def_plays=("EPA", "size"))
#     .reset_index()
# )

# off_epa_two_high = (
#     two_high
#     .groupby("OffTeam")
#     .agg(off_avg_epa=("EPA", "mean"),
#          off_plays=("EPA", "size"))
#     .reset_index()
#     .rename(columns={"OffTeam": "Team"})
# )

# print(off_epa_two_high.sort_values(by="off_avg_epa", ascending=False))
# def_epa_two_high = def_epa_two_high.rename(columns={"DefTeam": "Team"})

# team_two_high_epa = pd.merge(
#     def_epa_two_high,
#     off_epa_two_high,
#     on="Team",
#     how="inner"
# )

# corr = team_two_high_epa["def_avg_epa_allowed"].corr(
#     team_two_high_epa["off_avg_epa"]
# )

# print(f"Correlation: {corr:.3f}")

# plt.figure(figsize=(8, 6))
# plt.scatter(
#     team_two_high_epa["def_avg_epa_allowed"],
#     team_two_high_epa["off_avg_epa"]
# )

# for _, row in team_two_high_epa.iterrows():
#     plt.text(
#         row["def_avg_epa_allowed"] + 0.002,
#         row["off_avg_epa"] + 0.002,
#         row["Team"],
#         fontsize=9
#     )

# plt.xlim(-0.44, 0.44)
# plt.ylim(-0.44, 0.44)


# plt.axhline(0, alpha=0.3)
# plt.axvline(0, alpha=0.3)

# plt.xlabel("Defensive EPA Allowed vs Two-High")
# plt.ylabel("Offensive EPA vs Two-High")
# plt.title("Offense vs Defense Performance Against Two-High Coverages")
# plt.grid(alpha=0.1)
# plt.show()

##############################################
### Team Route Correlations w/ Z-Test ########
##############################################

# route_cols = ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"]

# two_high_coverages = [
#     "Cover 2",
#     "Cover 4",
#     "Cover 6",
#     "Cover 2 Man"
# ]

# not_two_high = df[
#     ~df["CoverageType"].isin(two_high_coverages)
# ].copy()

# routes_long_not_two_high = (
#     not_two_high
#     .melt(
#         id_vars=["OffTeam"],
#         value_vars=route_cols,
#         var_name="Alignment",
#         value_name="Route"
#     )
# )

# routes_long_not_two_high = routes_long_not_two_high.dropna(subset=["Route"])

# route_counts_not_two_high = (
#     routes_long_not_two_high
#     .groupby(["OffTeam", "Route"])
#     .size()
#     .reset_index(name="count")
# )

# route_counts_not_two_high["route_rate"] = (
#     route_counts_not_two_high["count"]
#     / route_counts_not_two_high.groupby("OffTeam")["count"].transform("sum")
# )

# route_counts_not_two_high = route_counts_not_two_high.sort_values(
#     ["OffTeam", "route_rate"],
#     ascending=[True, False]
# )

# pivot_table_not_two_high = route_counts_not_two_high.pivot(
#     index="OffTeam",
#     columns="Route",
#     values="route_rate"
# ).fillna(0)

# plt.figure(figsize=(12, 8))
# sns.heatmap(
#     pivot_table_not_two_high,
#     cmap="YlGnBu",
#     annot=True,
#     fmt=".2f",
#     cbar_kws={"label": "Route Usage Rate"}
# )

# plt.title("Route Usage Rates Against Non-Two-High Coverages")
# plt.xlabel("Route")
# plt.ylabel("Offensive Team")
# # plt.show()

# two_high = df[
#     df["CoverageType"].isin(two_high_coverages)
# ].copy()

# routes_long_two_high = (
#     two_high
#     .melt(
#         id_vars=["OffTeam"],
#         value_vars=route_cols,
#         var_name="Alignment",
#         value_name="Route"
#     )
# )

# routes_long_two_high = routes_long_two_high.dropna(subset=["Route"])

# route_counts_two_high = (
#     routes_long_two_high
#     .groupby(["OffTeam", "Route"])
#     .size()
#     .reset_index(name="count")
# )

# route_counts_two_high["route_rate"] = (
#     route_counts_two_high["count"]
#     / route_counts_two_high.groupby("OffTeam")["count"].transform("sum")
# )

# route_counts_two_high = route_counts_two_high.sort_values(
#     ["OffTeam", "route_rate"],
#     ascending=[True, False]
# )

# pivot_table_two_high = route_counts_two_high.pivot(
#     index="OffTeam",
#     columns="Route",
#     values="route_rate"
# ).fillna(0)

# route_diff = pivot_table_two_high - pivot_table_not_two_high

# plt.figure(figsize=(12, 8))
# sns.heatmap(
#     route_diff,
#     cmap="RdBu",
#     annot=True,
#     fmt=".2f",
#     cbar_kws={"label": "Route Usage Rate Difference"}
# )

# plt.title("Route Usage Rate Differences Between Two-High and Non-Two-High Coverages")
# plt.xlabel("Route")
# plt.ylabel("Offensive Team")
# # plt.show()

# two_high = ["Cover 2", "Cover 4", "Cover 6", "Cover 2 Man"]

# df["isTwoHigh"] = df["CoverageType"].isin(two_high).astype(int)

# df_stat_test = (
#     df
#     .melt(
#         id_vars=["OffTeam", "isTwoHigh"],
#         value_vars=route_cols,
#         var_name="Alignment",
#         value_name="Route"
#     )
# )

# df_stat_test = df_stat_test.dropna(subset=["Route"])

# route_counts = (
#     df_stat_test
#     .groupby(["OffTeam", "Route", "isTwoHigh"])
#     .size()
#     .unstack(fill_value=0)
#     .reset_index()
# )

# route_counts = (
#     df_stat_test
#     .groupby(["Route", "isTwoHigh"])
#     .size()
#     .unstack(fill_value=0)
# )

# total_routes = (
#     df_stat_test
#     .groupby("isTwoHigh")
#     .size()
# )

# results = []

# for route, row in route_counts.iterrows():
#     count = [row[1], row[0]]  # two-high, not-two-high
#     nobs = [total_routes[1], total_routes[0]]

#     if min(nobs) < 30 or min(count) < 5:
#         continue  # safety check

#     z_stat, p_value = proportions_ztest(count, nobs)

#     diff = (
#         row[1] / total_routes[1]
#         - row[0] / total_routes[0]
#     )

#     results.append({
#         "Route": route,
#         "two_high_rate": row[1] / total_routes[1],
#         "not_two_high_rate": row[0] / total_routes[0],
#         "rate_diff": diff,
#         "z_stat": z_stat,
#         "p_value": p_value
#     })

# route_significance = pd.DataFrame(results)

# print(route_significance.sort_values("p_value"))

##############################################
### Route and EPA Correlation in Two-High ####
##############################################

# route_cols = ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"]

# exclude_routes = ["Comeback"]

# two_high_coverages = [
#     "Cover 2",
#     "Cover 4",
#     "Cover 6",
#     "Cover 2 Man"
# ]

# df_two_high = df[
#     df["CoverageType"].isin(two_high_coverages)
# ].copy()

# route_cols = ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"]

# routes_long = (
#     df_two_high
#     .melt(
#         id_vars=["EPA"],
#         value_vars=route_cols,
#         var_name="Alignment",
#         value_name="Route"
#     )
#     .dropna(subset=["Route"])
# )

# route_epa = (
#     routes_long
#     .groupby("Route")
#     .agg(
#         avg_epa=("EPA", "mean"),
#         plays=("EPA", "size")
#     )
#     .reset_index()
# )

# route_epa = route_epa[~route_epa["Route"].isin(exclude_routes)]

# route_epa = route_epa[route_epa["plays"] >= 50]

# route_epa = route_epa.sort_values("avg_epa", ascending=False)

# plt.figure(figsize=(10, 6))
# plt.barh(
#     route_epa["Route"],
#     route_epa["avg_epa"],
# )

# plt.axvline(0, linestyle="--", alpha=0.5)
# plt.xlabel("Average EPA (Two-High Only)")
# plt.title("Routes Associated with EPA vs Two-High Coverages")
# plt.tight_layout()
# plt.show()

##############################################
### Formation Splits in Two-High #############
##############################################

# two_high_coverages = ["Cover 2", "Cover 4", "Cover 6", "Cover 2 Man"]

# df_two_high = df[df["CoverageType"].isin(two_high_coverages)].copy()
# df_not_two_high = df[~df["CoverageType"].isin(two_high_coverages)].copy()

# def canonical_split(row):
#     left = (row["LWR"], row["LSWR"], row["LTE"])
#     right = (row["RTE"], row["RSWR"], row["RWR"])

#     # Sort the two sides to remove left/right orientation
#     side1, side2 = sorted([left, right])

#     return f"{side1} | {side2}"

# df_two_high["canonical_split"] = df_two_high.apply(
#     canonical_split,
#     axis=1
# )

# split_counts = (
#     df_two_high["canonical_split"]
#     .value_counts()
#     .reset_index(name="count")
#     .rename(columns={"index": "canonical_split"})
# )

# split_counts["pct"] = (
#     split_counts["count"] / split_counts["count"].sum() * 100
# )

# split_counts = split_counts[split_counts["count"] > 30]

# df_not_two_high["canonical_split"] = df_not_two_high.apply(
#     canonical_split,
#     axis=1
# )

# split_counts_not = (
#     df_not_two_high["canonical_split"]
#     .value_counts()
#     .reset_index(name="count")
#     .rename(columns={"index": "canonical_split"})
# )

# split_counts_not["pct"] = (
#     split_counts_not["count"] / split_counts_not["count"].sum() * 100
# )

# #split_counts_not = split_counts_not[split_counts_not["count"] > 30]

# comparison = (
#     split_counts
#     .merge(
#         split_counts_not,
#         on="canonical_split",
#         suffixes=("_two_high", "_not_two_high"),
#         how="inner" 
#     )
# )

# comparison["pct_diff"] = (
#     comparison["pct_two_high"]
#     - comparison["pct_not_two_high"]
# )

# N_two_high = split_counts["count"].sum()
# N_not_two_high = split_counts_not["count"].sum()

# def two_proportion_z_test(row):
#     x1 = row["count_two_high"]
#     x2 = row["count_not_two_high"]
#     n1 = N_two_high
#     n2 = N_not_two_high

#     p1 = x1 / n1
#     p2 = x2 / n2
#     p_pool = (x1 + x2) / (n1 + n2)

#     se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
#     z = (p1 - p2) / se if se > 0 else 0
#     p_value = 2 * (1 - norm.cdf(abs(z)))

#     return pd.Series({"z": z, "p_value": p_value})

# comparison[["z", "p_value"]] = comparison.apply(
#     two_proportion_z_test,
#     axis=1
# )

# comparison = comparison[
#     (comparison["p_value"] < 0.001) &
#     (comparison["pct_diff"].abs() > 0.20)
# ]
# print(comparison)
# viz = comparison.sort_values("pct_diff")

# plt.figure(figsize=(8, 10))
# plt.barh(
#     viz["canonical_split"],
#     viz["pct_diff"]
# )
# plt.axvline(0)
# plt.xlabel("Usage Rate Difference (Two-High − Not Two-High)")
# plt.title("Formation Usage Shift vs Two-High Coverages")
# plt.tight_layout()
# plt.show()

#############################################
## Splits and EPA Correlation in Two-High ###
#############################################

# two_high_coverages = ["Cover 2", "Cover 4", "Cover 6", "Cover 2 Man"]

# df_two_high = df[df["CoverageType"].isin(two_high_coverages)].copy()
# df_not_two_high = df[~df["CoverageType"].isin(two_high_coverages)].copy()

# def canonical_split(row):
#     left = (row["LWR"], row["LSWR"], row["LTE"])
#     right = (row["RWR"], row["RSWR"], row["RTE"])

#     # Sort the two sides to remove left/right orientation
#     side1, side2 = sorted([left, right])

#     return f"{side1} | {side2}"

# df_two_high["canonical_split"] = df_two_high.apply(
#     canonical_split,
#     axis=1
# ) 

# split_two_high = (
#     df_two_high
#     .groupby("canonical_split")
#     .agg(
#         count=("EPA", "size"),
#         avg_yards=("YardsOnPlay", "mean"),
#         avg_epa=("EPA", "mean")
#     )
#     .reset_index()
# )

# split_two_high = split_two_high[split_two_high["count"] >= 50]

# top_yards = split_two_high.sort_values(
#     "avg_yards",
#     ascending=False
# )

# top_epa = split_two_high.sort_values(
#     "avg_epa",
#     ascending=False
# )

# split_two_high["yards_rank"] = split_two_high["avg_yards"].rank(ascending=False)
# split_two_high["epa_rank"] = split_two_high["avg_epa"].rank(ascending=False)

# split_two_high["composite_rank"] = (
#     split_two_high["yards_rank"] +
#     split_two_high["epa_rank"]
# )

# split_two_high = split_two_high.sort_values("yards_rank", ascending=False)

# plt.figure(figsize=(7, 6))

# plt.barh(
#     split_two_high["canonical_split"],
#     split_two_high["avg_yards"],
#     height=0.8,
#     color="coral",
#     alpha=0.7
# )
# plt.xlim(left=5)
# plt.xlabel("Average Yards per Play")
# plt.ylabel("Formation")
# plt.title("Formation Effectiveness vs Two-High Coverages")
# plt.tight_layout()
# # plt.show()


# route_cols = ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"]

# unique_routes = (
#     pd.unique(df[route_cols].values.ravel())
# )

# unique_routes = sorted(route for route in unique_routes if pd.notna(route))

# route_family = {
#     "Vertical": "vertical",
#     "Corner": "vertical",
#     "Post": "vertical",

#     "Out": "out",
    
#     "Comeback": "comeback",

#     "Hook": "hook",

#     "Flat": "flat",

#     "Cross": "cross",
#     "Shallow": "cross",

#     "Screen": "screen",
#     "Other": "other"
# }

# def normalize_route(route):
#     if pd.isna(route):
#         return None
#     return str(route).strip().title()

# def map_route_family(route):
#     route = normalize_route(route)
#     if route is None:
#         return None
#     return route_family.get(route, "other")


# def side_concept(row, cols):
#     families = []

#     for col in cols:
#         fam = map_route_family(row[col])
#         if fam is not None:
#             families.append(fam)

#     if not families:
#         return ()

#     return tuple(sorted(Counter(families).elements()))

# def full_route_concept(row):
#     left = side_concept(row, LEFT)
#     right = side_concept(row, RIGHT)

#     return tuple(sorted([left, right]))

# LEFT = ["L1", "L2", "L3", "L4"]
# RIGHT = ["R1", "R2", "R3", "R4"]

# two_high_coverages = ["Cover 2", "Cover 4", "Cover 6", "Cover 2 Man"]

# df = df[df["CoverageType"].isin(two_high_coverages)].copy()

# df["canonical_split"] = df.apply(
#     canonical_split,
#     axis=1
# )

# df["route_concept"] = df.apply(full_route_concept, axis=1)

# concept_usage = (
#     df
#     .groupby(["canonical_split", "route_concept"])
#     .agg(
#         count=("EPA", "size"),
#         avg_epa=("EPA", "mean")
#     )
#     .reset_index()
# )

# concept_usage["formation_total"] = (
#     concept_usage
#     .groupby("canonical_split")["count"]
#     .transform("sum")
# )

# concept_usage["pct"] = (
#     concept_usage["count"] / concept_usage["formation_total"]
# )

# concept_usage = concept_usage[concept_usage["formation_total"] >= 50]
# concept_usage = concept_usage[concept_usage["count"] >= 10]

# concept_usage = (
#     concept_usage
#     .sort_values(["canonical_split", "pct"], ascending=[True, False])
#     .groupby("canonical_split")
#     .head(10)
# )

# print(concept_usage)


#############################################
## Canonnical Splits and Route Combos #######
#############################################

two_high_coverages = ["Cover 2", "Cover 4", "Cover 6", "Cover 2 Man"]

df = df[df["CoverageType"].isin(two_high_coverages)].copy()

def canonical_split(row):
    left = (row["LWR"], row["LSWR"], row["LTE"])
    right = (row["RTE"], row["RSWR"], row["RWR"])

    return f"{left} | {right}"

df["CanonicalSplit"] = df.apply(canonical_split, axis=1)

route_family = {
    "Vertical": "vertical",
    "Corner": "vertical",
    "Post": "vertical",

    "Out": "out",
    
    "Comeback": "comeback",

    "Hook": "hook",

    "Flat": "flat",

    "Cross": "cross",
    "Shallow": "cross",

    "Screen": "screen",
    "Other": "other"
}

route_cols = ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"]

def simplify_route(route):
    if pd.isna(route):
        return route
    return route_family.get(route, "other")

for col in route_cols:
    df[col] = df[col].apply(simplify_route)

def concat_routes_with_split(row):
    left_counts = [int(row["LWR"]), int(row["LSWR"]), int(row["LTE"])]
    right_counts = [int(row["RTE"]), int(row["RSWR"]), int(row["RWR"])]

    left_route_order = ["L1", "L2", "L3", "L4"]
    right_route_order = ["R4", "R3", "R2", "R1"]

    left_routes = [row[c] for c in left_route_order if pd.notna(row[c])]
    right_routes = [row[c] for c in right_route_order if pd.notna(row[c])]


    def fill_side(counts, routes):
        out = []
        idx = 0
        for count in counts:
            if count == 0:
                out.append("0")
            else:
                assigned = routes[idx:idx + count]
                idx += count
                out.append("(" + ", ".join(assigned) + ")")
        return out

    left_out = fill_side(left_counts, left_routes)
    right_out = fill_side(right_counts, right_routes)


    return f"{', '.join(left_out)} | {', '.join(right_out)}"

df["RouteStructure"] = df.apply(concat_routes_with_split, axis=1)

def mirrored_alignment_route_structure(row):
    alignment = row["ReceiverAlignment"]

    # Collect routes
    left_routes = sorted(
        [row[c] for c in ["L1", "L2", "L3", "L4"] if pd.notna(row[c])]
    )
    right_routes = sorted(
        [row[c] for c in ["R1", "R2", "R3", "R4"] if pd.notna(row[c])]
    )

    left_tuple = tuple(left_routes)
    right_tuple = tuple(right_routes)

    original = (left_tuple, right_tuple)
    mirrored = (right_tuple, left_tuple)

    canonical_left, canonical_right = min(original, mirrored)

    left_str = (
        f"({', '.join(canonical_left)})"
        if canonical_left else "0"
    )
    right_str = (
        f"({', '.join(canonical_right)})"
        if canonical_right else "0"
    )

    return f"{alignment}: {left_str} | {right_str}"


df["AlignmentRouteStructure"] = df.apply(
    mirrored_alignment_route_structure,
    axis=1
)

def aggregate_by_alignment_and_routes(df):
    al_counts = (
        df.groupby("ReceiverAlignment")
          .size()
          .rename("al_count")
    )

    rt_counts = (
        df.groupby(["ReceiverAlignment", "AlignmentRouteStructure"])
          .size()
          .rename("rt_count")
          .reset_index()
    )

    out = rt_counts.merge(
        al_counts,
        on="ReceiverAlignment",
        how="left"
    )

    out["percentage"] = out["rt_count"] / out["al_count"] * 100

    performance = (
        df.groupby(["ReceiverAlignment", "AlignmentRouteStructure"])
          .agg(
              EPA=("EPA", "median"),
              YardsOnPlay=("YardsOnPlay", "median")
          )
          .reset_index()
    )
    out = out.merge(
        performance,
        on=["ReceiverAlignment", "AlignmentRouteStructure"],
        how="left"
    )
    return out

alignment_summary = aggregate_by_alignment_and_routes(df)

alignment_summary = alignment_summary[
    (alignment_summary["al_count"] > 50) &
    (alignment_summary["rt_count"] > 15)
]

print(alignment_summary.sort_values("EPA", ascending=False))

alignment_summary.to_csv(
    "alignment_route_summary.csv",
    index=False
)


### Sub analysis of exact split and route ########################

# def aggregate_by_split_and_route(df):
#     # --- count plays per canonical split ---
#     co_counts = (
#         df.groupby("CanonicalSplit")
#             .size()
#             .rename("co_count")
#     )


#     # --- count plays per (split, route structure) ---
#     ro_counts = (
#         df.groupby(["CanonicalSplit", "RouteStructure"])
#             .size()
#             .rename("ro_count")
#             .reset_index()
#     )


#     # --- merge split counts back in ---
#     out = ro_counts.merge(
#         co_counts,
#         on="CanonicalSplit",
#         how="left"
#     )

#     # --- compute percentage ---
#     out["percentage"] = out["ro_count"] / out["co_count"] * 100

#     # --- keep EPA and yards (aggregated meaningfully) ---
#     epa_yards = (
#     df.groupby(["CanonicalSplit", "RouteStructure"])
#         .agg(
#             EPA=("EPA", "mean"),
#             YardsOnPlay=("YardsOnPlay", "mean")
#             )
#             .reset_index()
#     )
#     out = out.merge(
#         epa_yards,
#         on=["CanonicalSplit", "RouteStructure"],
#         how="left"
#     )
#     return out

# summary_df = aggregate_by_split_and_route(df)

# summary_df = summary_df[
#     (summary_df["co_count"] > 40) &
#     (summary_df["ro_count"] > 5)
# ]

# # summary_df = summary_df[
# #     ~(
# #         ((summary_df["EPA"] < 0) & (summary_df["YardsOnPlay"] < 8)) |
# #         (summary_df["YardsOnPlay"] < 5)
# #     )
# # ]
# # print(summary_df.sort_values("EPA", ascending = False))

# def plot_top10_route_structures(df):
#     top10 = (
#         df.sort_values("YardsOnPlay", ascending=False)
#           .head(10)
#           .copy()
#     )

#     # Create readable labels
#     top10["label"] = (
#         top10["CanonicalSplit"].astype(str)
#         + "\n"
#         + top10["RouteStructure"]
#     )

#     # Plot
#     plt.figure(figsize=(10, 6))
#     plt.barh(top10["label"], top10["YardsOnPlay"], color="teal", alpha=0.7)
#     plt.xlabel("Mean Yards On Play")
#     plt.title("Top 10 Route Structures by Yards On Play\n(Conditioned on Canonical Split)")
#     plt.gca().invert_yaxis()
#     plt.tight_layout()
#     plt.show()

# # plot_top10_route_structures(summary_df)


# top10_global = (
#     summary_df
#     .sort_values("EPA", ascending=False)
#     .head(10)
# )

# # Extract the CanonicalSplits involved
# top10_splits = top10_global["CanonicalSplit"].unique()

# subset_df = summary_df[
#     summary_df["CanonicalSplit"].isin(top10_splits)
# ]

# subset_df = subset_df.copy()

# subset_df["epa_rank_within_split"] = (
#     subset_df
#     .groupby("CanonicalSplit")["EPA"]
#     .rank(method="first", ascending=False)
# )

# top3_per_split = subset_df[
#     subset_df["epa_rank_within_split"] <= 3
# ].sort_values(
#     ["CanonicalSplit", "epa_rank_within_split"]
# )
# # print(top3_per_split)

# target_route_structure = "0, (cross, cross), 0 | 0, (out), (vertical)"

# filtered_plays = df[
#     df["RouteStructure"] == target_route_structure
# ]

##############################################
### Named Formation Types and L/R splits #####
##############################################

# two_high_counts = (
#     two_high.groupby("ReceiverAlignment")
#       .size()
#       .rename("total_count")
# )

# alignment_two_high_dist = two_high_counts.reset_index()
# alignment_two_high_dist["percentage"] = (
#     (alignment_two_high_dist["total_count"] /
#     alignment_two_high_dist["total_count"].sum()) * 100
# )

# not_two_high_counts = (
#     not_two_high.groupby("ReceiverAlignment")
#       .size()
#       .rename("total_count")
# )

# alignment_not_two_high_dist = not_two_high_counts.reset_index()
# alignment_not_two_high_dist["percentage"] = (
#     (alignment_not_two_high_dist["total_count"] /
#     alignment_not_two_high_dist["total_count"].sum()) * 100
# )

# print(alignment_two_high_dist.sort_values("percentage", ascending=False))
# print(alignment_not_two_high_dist.sort_values("percentage", ascending=False))


# comp_df = alignment_two_high_dist.merge(
#     alignment_not_two_high_dist,
#     on="ReceiverAlignment",
#     suffixes=("_two", "_not"),
#     how="inner"
# )

# comp_df["pct_diff"] = (
#     comp_df["percentage_two"] - comp_df["percentage_not"]
# )

# n_two = alignment_two_high_dist["total_count"].sum()
# n_not = alignment_not_two_high_dist["total_count"].sum()

# def two_proportion_z_test(x1, n1, x2, n2):
#     p1 = x1 / n1
#     p2 = x2 / n2
#     p_pool = (x1 + x2) / (n1 + n2)

#     se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
#     z = (p1 - p2) / se
#     p_value = 2 * (1 - norm.cdf(abs(z)))

#     return z, p_value

# results = []

# for _, row in comp_df.iterrows():
#     z, p = two_proportion_z_test(
#         row["total_count_two"], n_two,
#         row["total_count_not"], n_not
#     )

#     results.append((z, p))

# comp_df["z_score"] = [r[0] for r in results]
# comp_df["p_value"] = [r[1] for r in results]

# comp_df["significant_5pct"] = comp_df["p_value"] < 0.01

# summary_df = comp_df[
#     (comp_df["significant_5pct"] == True) &
#     (comp_df["pct_diff"].abs() > 0.5)
# ]

# print(summary_df.sort_values("pct_diff", ascending=False))