# FrontFin - Financial Advisor Bot (CM3070 - Final Project)
This is the GitHub repo for FrontFin, a financial advisor bot developed as part of my CM3070 Final Project. It includes code for data preprocessing, stock analysis, and portfolio recommendation via a user-friendly web app.

1. For necessary dependencies and libraries installtion, please run this command before running the project in your terminal:
pip install pandas numpy yfinance matplotlib seaborn fastapi uvicorn scikit-learn tqdm tabulate pydantic statsmodels plotly pyarrow fastparquet

2. Locate to the folder of where you kept FrontFin, and if your folder name is cm3070-final-project as you downloaded from here, please type and run this in your terminal:
cd cm3070-final-project

3. After dependencies and libraries are installed, then run the app in your terminal:
python -m uvicorn app:app --reload
OR
uvicorn app:app --reload

