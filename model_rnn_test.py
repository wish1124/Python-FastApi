import pandas as pd
from model_rnn import run_training_rnn

def main():
    df = pd.read_csv("test_bid_4features.csv")
    res = run_training_rnn(
        df=df,
        feature_cols=("기초금액", "추정가격", "예가범위", "낙찰하한율"),
        target_col="낙찰가",
        target_log=True,
        epochs=50,
        patience=10,
        batch_size=256,
        lr=1e-3,
    )
    print("Best VAL:", res.best_val)
    print("TEST:", res.test)

if __name__ == "__main__":
    main()
