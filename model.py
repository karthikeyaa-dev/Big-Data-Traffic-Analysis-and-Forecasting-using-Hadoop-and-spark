# ==========================================================
# FAST CUDA ST-GCN Traffic Forecasting
# ==========================================================

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch_geometric.nn import GCNConv

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tqdm.auto import tqdm



# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))



# ==========================================================
# 1. GRAPH LOADING
# ==========================================================

graph_path = "/kaggle/input/datasets/karthikeyaa6274/t2raffic-data-v1/graph_csv.csv"


graph_df = pd.read_csv(
    graph_path,
    dtype={
        "source_sensor":"int32",
        "destination_sensor":"int32",
        "weight":"float32"
    }
)



sensors = np.unique(
    np.concatenate(
        [
            graph_df.source_sensor.values,
            graph_df.destination_sensor.values
        ]
    )
)


sensor_to_id = {
    s:i for i,s in enumerate(sensors)
}


num_nodes=len(sensors)


print(
    "Sensors:",
    num_nodes
)



edge_index=torch.tensor(
    [
        graph_df.source_sensor.map(sensor_to_id).values,
        graph_df.destination_sensor.map(sensor_to_id).values
    ],
    dtype=torch.long,
    device=device
)



edge_weight=torch.tensor(
    graph_df.weight.values,
    dtype=torch.float32,
    device=device
)



print(
    "Edges:",
    edge_index.shape[1]
)



# ==========================================================
# 2. LOAD FEATURE DATA
# ==========================================================

feature_path="/kaggle/input/datasets/karthikeyaa6274/t2raffic-data-v1/gnn_features_csv.csv"


df=pd.read_csv(
    feature_path
)


print(
    "Rows:",
    len(df)
)



df=df[
    df.sensor_id.isin(sensors)
]


df["timestamp"]=pd.to_datetime(
    df["timestamp"]
)



# ==========================================================
# 3. FEATURE CLEANING
# ==========================================================


drop_cols=[
    "sensor_id",
    "timestamp",
    "traffic_trend_vector"
]



for c in df.columns:

    if c not in drop_cols:

        df[c]=pd.to_numeric(
            df[c],
            errors="coerce"
        )



df=df.fillna(0)



feature_cols=[
    c for c in df.columns
    if c not in [
        "target_speed"
    ]
    and c not in drop_cols
    and np.issubdtype(
        df[c].dtype,
        np.number
    )
]



print(
    "Features:",
    len(feature_cols)
)



# ==========================================================
# 4. SCALE FEATURES
# ==========================================================


scaler=StandardScaler()


df[feature_cols]=scaler.fit_transform(
    df[feature_cols]
)



# ==========================================================
# 5. FAST SNAPSHOT CREATION
# ==========================================================


print(
    "Building snapshots..."
)



df["node_id"]=df.sensor_id.map(
    sensor_to_id
)



feature_values=df[
    feature_cols
].values.astype(
    np.float32
)



if "target_speed" in df.columns:

    target_values=df.target_speed.values.astype(
        np.float32
    )

else:

    target_values=df.speed.values.astype(
        np.float32
    )



nodes=df.node_id.values.astype(
    np.int32
)



time_values=df.timestamp.values



unique_times, inverse=np.unique(
    time_values,
    return_inverse=True
)



num_times=len(unique_times)



print(
    "Time steps:",
    num_times
)



X_snap=np.zeros(
    (
        num_times,
        num_nodes,
        len(feature_cols)
    ),
    dtype=np.float32
)



Y_snap=np.zeros(
    (
        num_times,
        num_nodes
    ),
    dtype=np.float32
)



for i in tqdm(
    range(num_times),
    desc="Snapshots"
):

    mask=inverse==i

    n=nodes[mask]


    X_snap[i,n]=feature_values[mask]

    Y_snap[i,n]=target_values[mask]



print(
    "Snapshot shape:",
    X_snap.shape
)



# ==========================================================
# 6. DATASET
# ==========================================================


SEQ_LEN=12



class TrafficDataset(Dataset):


    def __init__(
        self,
        X,
        Y
    ):

        self.X=X
        self.Y=Y



    def __len__(self):

        return len(self.X)-SEQ_LEN



    def __getitem__(
        self,
        idx
    ):

        return (

            torch.from_numpy(
                self.X[
                    idx:idx+SEQ_LEN
                ]
            ),

            torch.from_numpy(
                self.Y[
                    idx+SEQ_LEN
                ]
            )

        )



split=int(
    num_times*0.8
)



train_dataset=TrafficDataset(
    X_snap[:split],
    Y_snap[:split]
)



val_dataset=TrafficDataset(
    X_snap[split:],
    Y_snap[split:]
)



train_loader=DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True,
    num_workers=2
)



val_loader=DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=True,
    num_workers=2
)



# ==========================================================
# 7. MODEL
# ==========================================================


class STGCN(nn.Module):


    def __init__(
        self,
        input_dim
    ):

        super().__init__()


        self.gcn1=GCNConv(
            input_dim,
            64
        )


        self.gcn2=GCNConv(
            64,
            32
        )


        self.gru=nn.GRU(
            32,
            64,
            batch_first=True
        )


        self.fc=nn.Linear(
            64,
            1
        )



    def forward(
        self,
        x
    ):


        batch,time,nodes,features=x.shape


        outputs=[]


        for t in range(time):


            xt=x[:,t]


            h=self.gcn1(
                xt.reshape(
                    -1,
                    features
                ),
                edge_index,
                edge_weight
            )


            h=F.relu(h)


            h=self.gcn2(
                h,
                edge_index,
                edge_weight
            )


            h=h.reshape(
                batch,
                nodes,
                -1
            )


            outputs.append(h)



        h=torch.stack(
            outputs,
            dim=1
        )



        h=h.permute(
            0,
            2,
            1,
            3
        )



        h=h.reshape(
            batch*nodes,
            time,
            -1
        )



        h,_=self.gru(
            h
        )



        h=h[:,-1]



        out=self.fc(
            h
        )


        return out.reshape(
            batch,
            nodes
        )




model=STGCN(
    len(feature_cols)
).to(device)



optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)



criterion=nn.HuberLoss()



# ==========================================================
# 8. TRAIN WITH CUDA AMP
# ==========================================================


epochs=30


amp_scaler=torch.cuda.amp.GradScaler()



for epoch in range(
    epochs
):


    model.train()


    running=0



    loop=tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{epochs}"
    )


    for X,y in loop:


        X=X.to(
            device,
            non_blocking=True
        )


        y=y.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad()



        with torch.cuda.amp.autocast():

            pred=model(
                X
            )

            loss=criterion(
                pred,
                y
            )



        amp_scaler.scale(
            loss
        ).backward()



        amp_scaler.step(
            optimizer
        )


        amp_scaler.update()



        running+=loss.item()


        loop.set_postfix(
            loss=loss.item()
        )



    print(
        "Train loss:",
        running/len(train_loader)
    )



# ==========================================================
# 9. EVALUATION
# ==========================================================


model.eval()


preds=[]
actual=[]



with torch.no_grad():

    for X,y in val_loader:


        X=X.to(device)


        pred=model(
            X
        )


        preds.append(
            pred.cpu().numpy()
        )


        actual.append(
            y.numpy()
        )



preds=np.concatenate(
    preds
)


actual=np.concatenate(
    actual
)



rmse=np.sqrt(
    mean_squared_error(
        actual.flatten(),
        preds.flatten()
    )
)



mae=mean_absolute_error(
    actual.flatten(),
    preds.flatten()
)



print("\nFINAL RESULTS")
print("RMSE:",rmse)
print("MAE :",mae)



result=pd.DataFrame(
    {
        "actual":actual.flatten(),
        "predicted":preds.flatten()
    }
)


print(
    result.head(20)
)
