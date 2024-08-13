from tbparse import SummaryReader
import pandas as pd
import os

energy_cols = ['timestamp',
 'RAM%',
 'GPU%',
 'GPU inst power (mW)',
 'GPU avg power (mW)',
 'CPU%',
 'CPU inst power (mW)',
 'CPU avg power (mW)',
 'tot inst power (mW)',
 'tot avg power (mW)']


def convert_tb_to_csv(log_dir: str, csv_path: str):
    reader = SummaryReader(log_dir)
    df = reader.scalars
    df.to_csv(csv_path, index=False)
    return csv_path


def merge_results(log_dir, exp_summary_path):
    exp_summary = pd.read_csv(exp_summary_path)
    tensorboard = pd.DataFrame()
    energy = pd.DataFrame()
    for index, data in exp_summary.iterrows():
        date_exp = data["date_exp"]
        print(f"Processing {date_exp}")
        # training logs
        if not os.path.exists(os.path.join(log_dir, date_exp)):
            print(f"Directory not found for {date_exp}")
            exp_summary.drop(index, inplace=True)
            continue
        tb_path = os.path.join(log_dir, date_exp, "tb.csv")
        if not os.path.exists(tb_path):
            try:
                convert_tb_to_csv(os.path.join(log_dir, date_exp), tb_path)
            except Exception as e:
                print(f"Tensorboard log not found for {date_exp} with error: {e}")
                exp_summary.drop(index, inplace=True)
            continue
        try:
            tb = pd.read_csv(tb_path)
        except Exception as e:
            print("Reading the tensorboard csv failed with error: %s", e)
            exp_summary.drop(index, inplace=True)
            continue
        tb["metric"], tb["phase"] = tb["tag"].apply(lambda x: x.split("/")[0]), tb["tag"].apply(lambda x: x.split("/")[1] if len(x.split("/")) > 1 else "no_phase")
        tb.drop(columns=["tag"], inplace=True)
        tb["date_exp"] = date_exp
        if tb[tb["metric"] == "Loss"].shape[0] <= 1:
            continue
        
        # energy monitoring
        energy_path = os.path.join(log_dir, date_exp, "energy.csv")
        try:
            nrg = pd.read_csv(energy_path)
        except:
            print(f"Energy log not found for {date_exp}")
            exp_summary.drop(index, inplace=True)
            continue
        nrg["date_exp"] = date_exp
        
        # if both logs are found and correct, append to the dataframes
        tensorboard = pd.concat([tensorboard, tb], join="outer")
        energy = pd.concat([energy, nrg], join="outer")
    energy = energy.merge(exp_summary, on="date_exp")
    tensorboard = tensorboard.merge(exp_summary, on="date_exp")
    return exp_summary,tensorboard, energy

if __name__ == "__main__":
    log_dir = "/home/mjay/ai-energy-consumption/Jetson/logs"
    log_dir = "/Users/mathildepro/Documents/code_projects/ai-energy-consumption-framework/Jetson/logs"
    dirs = [x for x in os.listdir(log_dir) if "." not in x]
    print(dirs)
    for d in dirs:
        tb_csv = os.path.join(log_dir, d, "tb.csv")
        if not os.path.exists(tb_csv):
            convert_tb_to_csv(os.path.join(log_dir, d), tb_csv)