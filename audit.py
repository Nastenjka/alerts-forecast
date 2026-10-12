import pandas as pd

def rd(p):
    d = pd.read_csv(p)
    for c in ("started_at", "finished_at"):
        d[c] = pd.to_datetime(d[c], utc=True)
    d["dur_min"] = (d["finished_at"] - d["started_at"]).dt.total_seconds() / 60
    d["month"] = d["started_at"].dt.strftime("%Y-%m")
    return d

v = rd("data/volunteer_data_en.csv")
o = rd("data/official_data_en.csv")

print("VOLUNTEER regions:\n", v["region"].value_counts().to_string())
print("\nOFFICIAL level:\n", o["level"].value_counts())
print("OFFICIAL oblasts:", o["oblast"].nunique())

print("\nDuration (min):\n", v["dur_min"].describe())
print("negative:", (v.dur_min < 0).sum(), "| zero:", (v.dur_min == 0).sum(),
      "| >24h:", (v.dur_min > 1440).sum())
print(v.nlargest(8, "dur_min")[["region", "started_at", "dur_min", "naive"]])

print("\nnaive share overall:", v["naive"].mean().round(3))
print("naive == exactly 30 min:", (v.loc[v.naive, "dur_min"] == 30).mean().round(3))
print("non-naive exactly 30 min:", (v.loc[~v.naive, "dur_min"] == 30).sum())
print("naive share, last 12 months:\n", v.groupby("month")["naive"].mean().tail(12).round(3))
print("naive share by region:\n", v.groupby("region")["naive"].mean().round(3).sort_values().to_string())

print("\nduplicate (region, start):", v.duplicated(["region", "started_at"]).sum())
vs = v.sort_values(["region", "started_at"])
print("overlaps within region:", (vs["started_at"] < vs.groupby("region")["finished_at"].shift()).sum())

print("\nOFFICIAL level by month (Jun 2025 onward):")
print(pd.crosstab(o["month"], o["level"]).loc["2025-06":].to_string())

d = pd.concat({
    "volunteer": v.set_index("started_at").resample("D").size(),
    "official": o.set_index("started_at").resample("D").size(),
}, axis=1)
print("\nDaily alert counts, last 60 days:\n", d.tail(60).to_string())

om = o[o["level"] == "oblast"].groupby("month").size()
print("\nMonthly counts (volunteer vs official oblast-level):")
print(pd.concat({"volunteer": v.groupby("month").size(), "official_oblast": om}, axis=1).tail(24).to_string())