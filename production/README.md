# Production
How to make the model into a product

## Web application

1. Scrutinize scanned document and remind user if missing / abnormal;

2. Generate recommendation on approval / suggestions on default or credit rating;

3. Collect feedback from user to improve current model

Why not active learning: active learning pick out the most import _unlabelled_ sample for user to label and hence gets better and better. But in our case, the wrongly labelled sample might just be an exception.
