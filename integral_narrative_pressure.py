import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./reports/AHOI/timeline.csv")

df["cum_energy"] = df["z_energy_global"].cumsum()

plt.figure(figsize=(12,4))
plt.plot(df.index, df["cum_energy"])
plt.title("Cumulative Narrative Energy")
plt.xlabel("Scene")
plt.ylabel("Cumulative z_energy")
plt.show()
