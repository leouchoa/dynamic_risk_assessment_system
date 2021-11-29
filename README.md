# Dynamic Risk Assessment System
## Background

Imagine that you're the Chief Data Scientist at a big company that has 10,000 corporate clients. Your company is extremely concerned about attrition risk: the risk that some of their clients will exit their contracts and decrease the company's revenue. They have a team of client managers who stay in contact with clients and try to convince them not to exit their contracts. However, the client management team is small, and they're not able to stay in close contact with all 10,000 clients.

The company needs you to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. If the model you create and deploy is accurate, it will enable the client managers to contact the clients with the highest risk and avoid losing clients and revenue.

Creating and deploying the model isn't the end of your work, though. The industry is dynamic and constantly changing, and a model that was created a year or a month ago might not still be accurate today. Because of this, it is needed to set up regular monitoring of the model to ensure that it remains accurate and up-to-date. 

This project touches upon setting up processes and scripts to re-train, re-deploy, monitor, and report on your ML model, so that your company can get risk assessments that are as accurate as possible and minimize client attrition.

## Steps

The main steps to have a sucessful monitoring of the model are:

- Ingestion
- Training
- Scoring
- Deployment
- Diagnostics
- Reporting

With those steps we can better assess our model performance.

The overall high level idea of how those steps work together is depicted in figure bellow.

![](fullprocess.jpg)

### Diagnostics

The diagnostics step is where we check if our model is suffering from [model drift](https://datatron.com/what-is-model-drift/), indicating that its perfomance is as good as it once was. Here the diagnostic is done by comparing a current [f1-score](https://deepai.org/machine-learning-glossary-and-terms/f-score) with the f1-score given by scoring the model against new data. If the new score is less (raw comparison test for now) than the current, it is considered that model drift has happened.


### Ingestion

This step ingests multiple data from a source directory and checks whether there's at least one file not already ingested. If that's the case then the data is again ingested and compiled to form a bigger dataset for next steps.

### Training

Model training will happen if the diagnostics step reports that has occurred
