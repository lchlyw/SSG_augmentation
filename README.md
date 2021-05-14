# SSG_augmentation
Shematic diagram of SSG for problem-specific dataset augmentation.

![Schematic flow chart of SSG system](https://user-images.githubusercontent.com/44221597/118208147-34975780-b4a1-11eb-88d5-fc38db305e6b.png)

Step 1: Train and predict new variable using machine learning model. 
Step 2: Filter "bad" prediction using statistical feature relation.
Step 3: Update the dataset. Cycle continues until desired amount of dataset is achieved.
