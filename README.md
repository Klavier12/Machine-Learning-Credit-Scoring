# Credit Score Analysis

## Index
1. [Introduction](#introduction)
2. [Problem Context](#problem-context)
3. [Project Aim](#project-aim)
4. [Variables Used](#variables-used)
5. [Project Description](#project-description)
6. [Repository Structure](#repository-structure)
7. [Objectives](#objectives)
8. [Key Performance Indicators (KPIs)](#key-performance-indicators-kpis)
9. [Expected Results](#expected-results)
10. [Support Material](#support-material)

## Introduction
Credit scoring is a crucial tool in the financial sector, helping institutions assess the creditworthiness of individuals. This project focuses on developing a robust credit scoring system using Machine Learning models. By leveraging historical financial data, the project aims to predict whether a loan applicant is likely to default or not. The implementation includes advanced algorithms optimized for accuracy and scalability, ensuring financial institutions can make informed lending decisions.

## Problem Context
Financial institutions rely heavily on credit scores to determine loan eligibility and mitigate risks. Traditional credit scoring models often lack adaptability, leading to outdated assessments and financial losses. The integration of Machine Learning in credit evaluation provides a data-driven, dynamic approach that enhances decision-making. Accurate credit scoring helps financial entities identify applicants who are more likely to meet their financial obligations, thereby reducing the risk of bad debt and maintaining financial health.

## Project Aim
The primary objective of this project is to develop a credit scoring model that allows financial institutions to assess the likelihood of an applicant meeting their financial obligations. This ensures responsible lending practices and minimizes potential financial risks.

## Variables Used
The dataset includes the following key variables:
- **Age**: Applicant's age.
- **Income**: Monthly income level
- **Credit History**: Past credit performance and defaults.
- **Loan Amount**: Amount requested by the applicant.
- **Employment Status**: Type of employment and stability.
- **Debt-to-Income Ratio**: Percentage of income allocated to existing debts.
- **Number of Open Accounts**: Active credit lines.
- **Previous Defaults**: History of missed payments.

## Project Description
This project involves training and evaluating two Machine Learning models: **Random Forest** and **CatBoost**. The models were optimized using **GridSearch** and **Cross-Validation** techniques to improve performance and accuracy. The dataset used consists of financial records that help in predicting the creditworthiness of applicants.

### Technologies Used:
- Python
- Scikit-Learn
- CatBoost
- Pandas & NumPy
- Matplotlib & Seaborn
- GridSearchCV & Cross-Validation
- Docker
- FastAPI

## Repository Structure
```
├── data/                 # Dataset and preprocessed data
├── notebooks/            # Jupyter Notebooks with data analysis and model training
├── src/                  # Source code for data processing and model training
├── api/                  # FastAPI implementation for model deployment
├── Dockerfile            # Configuration for containerized deployment
├── requirements.txt      # List of required dependencies
├── README.md             # Project documentation
```

## Objectives
1. Develope an effective credit scoring model using Machine Learning.
2. Compare the performance of Random Forest and CatBoost models.
3. Optimize models using GridSearch and Cross-Validation.
4. Deploye the best-performing model using FastAPI and Docker.
5. Provide an accessible and scalable solution for financial institutions.

## Key Performance Indicators (KPIs)
- **Accuracy**: Measure of correctly classified loan applicants.
- **Precision & Recall**: Evaluate the model’s ability to detect defaults correctly.
- **ROC-AUC Score**: Assess the model’s ability to distinguish between defaulters and non-defaulters.
- **F1 Score**: Balance between precision and recall to ensure optimal performance.

## Expected Results
By the end of this project, we expect to:
- Achieve a highly accurate credit scoring model.
- Provide insights into key financial risk factors.
- Develop an API for real-time credit evaluation.
- Ensure model scalability through Docker deployment.

### Model Performance
The best-performing model achieved an **85% ROC-AUC score**, which indicates a strong ability to distinguish between applicants who are likely to default and those who are not. The **ROC-AUC score** is a key metric because it evaluates how well the model separates positive from negative cases, making it ideal for credit risk analysis.

## Support Material
- [Machine Learning for Credit Scoring](https://www.sciencedirect.com/topics/computer-science/credit-scoring)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)


