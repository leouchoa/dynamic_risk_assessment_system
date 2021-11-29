# Next Steps
## PDF reports

In Step 4 "Reporting", you set up a script that generates a plot of a confusion matrix. Instead of outputting just that raw plot, set up a script that generates a pdf file that contains the plot as well as summary statistics and other diagnostics. This enables more complete, quicker reporting that will really make your project stand out.

In order to accomplish this suggestion, you'll need to add to your reporting.py Python script. You may also need to install modules that enable PDF creation, such as the reportlab module. There are many things you could include in a PDF report about your model: you could include the confusion matrix you generate in reporting.py, you could include all of the outputs of API endpoints you created in app.py, and you could also include the model's F1 score (stored in latestscore.txt) and the files that you ingested to train the model (stored in ingestedfiles.txt).
## Time Trends

Give your scripts the ability to store diagnostics from previous iterations of your model, and generate reports about time trends. For example, show how the percent of NA elements has gone up or down over many weeks or months, or show whether the timing of ingestion and training has increased or decreased.

You could accomplish this suggestion in several different ways. For example, you could create a directory called /olddiagnostics/, and create a script that copied all of your diagnostics outputs to that folder. You could also add timestamps to the filenames of your output files like ingestedfiles.txt and latestscore.txt.
## Database setup

Instead of writing results and files to .txt and .csv files, write your datasets and records to SQL databases. This will lead to increased performance and reliability.

In order to accomplish this suggestion, you'll have to set up SQL databases in your workspace. You can accomplish this within Python by installing and using the mysql-connector-python module. You could create a new Python script called dbsetup.py that used this module to set up databases. You could set up a database that stored information about ingested files, another one to store information about model scores, and another one to store information about model diagnostics. Then, you would have to alter ingestion.py, scoring.py, and diagnostics.py so that they wrote to these databases every time they were run.

