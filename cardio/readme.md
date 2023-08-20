## Cardiovascular Diseases Risk Prediction Dataset (The 2021 BRFSS Dataset from CDC)
Instruction: Extract the zip file for the dataset

In the dataset, many interpretations can be done. 

### 1. general health.py
The first Python file checks if such lifestyle and their current state can affect the person's General Health.
Using a classification algorithm (Random Forest), I will determine how the model interprets it.

After feeding the model with the data and keeping only the necessary data, there were several things that greatly influenced one's general health.
The model achieved 95% accuracy in the 80-20 ratio for training and testing, respectfully. The ratio is kept at a level as the data that was used is quality grade.

The model also had a recall value of 1, which tells how the model can be perfect for detecting a person's general health correctly. 

### 2. heart disease.py
Like the general health method, this time it will check how such a lifestyle affects one's condition.
