import numpy as np
import pandas as pd

def load_data(path : str) -> pd.DataFrame:
    sheet = pd.read_excel(path)

    df = pd.DataFrame({
        "Energy Confinement Time":sheet["TAUTOT]"][0:-1], # energy confinement time
        "Magnetic Flux Density":sheet["BT"][0:-1],
        "Plasma Current":sheet["IP"][0:-1], #Plasma current
        "Thermal Power":sheet["PLTH"][0:-1], #thermal power
        "Major Radius":sheet["RGEO"][0:-1], # major radius
        "Elongation":sheet["KAPPA"][0:-1], #elongation
        "Electron Density":sheet["NEL"][0:-1], #electron density
        "Effective Mass Number": sheet["MEFF"][0:-1], #effective mass number
        "Inverse Aspect Ratio": sheet["AMIN"][0:-1]/sheet["RGEO"][0:-1], # inverse aspect ratio
        "Tokamak Name": sheet["TOK"][0:-1], #tokamak name
        "Wall Material": sheet["WALMAT"][0:-1], #material of surrounding walls
        "Phase": sheet["PHASE"][0:-1], #what confinement mode the plasma is in
    })

    df["Magnetic Flux Density"] = np.abs(df["Magnetic Flux Density"])
    #ensure all data is numerical and no NANs
    df = df.dropna(axis=0)
    df['Electron Density'] = df['Electron Density'].astype(float)
    # pick data corresponding to the JET tokamak
    df = df[df['Tokamak Name']=="JET"]
    #select only data with a carbon wall
    df = df.loc[["C" in entry for entry in df['Wall Material']]]
    #select data only in high-confinement mode, or H-mode
    df = df.loc[[((entry=="H") or (entry=="HSELM") or (entry=="HGELM")) for entry in df['Phase']]]


    return df


