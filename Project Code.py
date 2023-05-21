
import json
import pandas as pd
from striprtf.striprtf import rtf_to_text
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def process_iris_dataset(json_file_path, iris_dataset_path):
    # Read the JSON file and strip RTF
    with open(json_file_path, 'r') as file:
        json_content = file.read()
        json_content = rtf_to_text(json_content)
        instructions = json.loads(json_content)

    # Print target feature and type of regression
    target_feature = instructions["design_state_data"]["target"]["target"]
    model_type = instructions["design_state_data"]["target"]["target"]
    print(f"Target Feature: {target_feature}")
    print(f"Regression Type: {model_type}")

    # Load the Iris dataset
    iris_df = pd.read_csv(iris_dataset_path)

    # Impute missing values and perform tokenization and hashing

    feature_name = {}
    num_of_columns = len(iris_df.columns)
    print("Number of feature to be handled: ", num_of_columns)
    feature_handling_data = instructions["design_state_data"]["feature_handling"]
    for i in range(0, num_of_columns):
        feature_name[i] = feature_handling_data[iris_df.columns[i]]
        if feature_name[i]["feature_variable_type"] == "numerical":
            if feature_name[i]["feature_details"]["numerical_handling"] == "Keep as regular numerical feature":
                feature_name[i]["feature_details"].pop("numerical_handling")
            if feature_name[i]["feature_details"]["rescaling"] == "No rescaling":
                feature_name[i]["feature_details"].pop("rescaling")
            if feature_name[i]["feature_details"]["make_derived_feats"] == False:
                feature_name[i]["feature_details"].pop("make_derived_feats")
            if feature_name[i]["feature_details"]["missing_values"] != "Impute":
                feature_name[i]["feature_details"].pop("missing_values", "impute_with", "impute_value")
        elif feature_name[i]["feature_variable_type"] == "text":
            if feature_name[i]["feature_details"]["text_handling"] != "Tokenize and hash":
                feature_name[i]["feature_details"].pop("text_handling", "hash_columns")
        print("Feature Name:", feature_name[i]["feature_name"], "\nHandling: ", feature_name[i]["feature_details"])

    imputation_pipeline_steps = []
    tokenization_pipeline_steps = []

    for i in range(0, num_of_columns):
        if feature_name[i]["feature_variable_type"] == "numerical":
            print(feature_name[i]['feature_name'])
            if feature_name[i]["feature_details"]["impute_with"] == "Average of values":
                imputer = SimpleImputer(strategy="mean")
            elif feature_name[i]["feature_details"]["impute_with"] == "custom":
                imputer = SimpleImputer(strategy="constant", fill_value=feature_name[i]["feature_details"]["impute_value"])

            imputation_pipeline_steps.append((feature_name[i]["feature_name"], imputer))
            
        elif feature_name[i]["feature_variable_type"] == "text":
            print(feature_name[i]['feature_name'])
            
            tokenizer = HashingVectorizer(n_features=10, alternate_sign=False)
            tokenization_pipeline_steps.append((feature_name[i]["feature_name"], tokenizer))
            print(tokenization_pipeline_steps)
    imputation_pipeline = Pipeline(steps=imputation_pipeline_steps)
    X_imputed = imputation_pipeline.fit_transform(iris_df.iloc[:,:4])

    tokenization_pipeline = Pipeline(steps=tokenization_pipeline_steps)
    species_text = iris_df['species'].astype(str)
    X_tokenized = tokenization_pipeline.fit_transform(species_text)
    print(X_tokenized.shape)
    
    
    # Feature reduction
    feature_reduction_method = instructions["design_state_data"]["feature_reduction"]["feature_reduction_method"]
    k = int(instructions["design_state_data"]["feature_reduction"]["num_of_features_to_keep"])

    if feature_reduction_method == 'No Reduction':
        print("No feature reduction required")
    elif feature_reduction_method == 'Correlation with target':
        feature_reduction_estimator = SelectKBest(score_func=f_regression, k=k)
    elif feature_reduction_method == 'Tree-based':
        
        #n_estimators = int(instructions["design_state_data"]["feature_reduction"]["num_of_trees"])
        #max_depth = int(instructions["design_state_data"]["feature_reduction"]["depth_of_trees"])
        feature_reduction_estimator = SelectKBest(score_func=ExtraTreesClassifier, k=k)
    elif feature_reduction_method == 'Principal Component Analysis':
        feature_reduction_estimator = PCA(n_components=k)
    else:
        raise ValueError(f"Invalid feature reduction method: {feature_reduction_method}")

    feature_reduction_pipeline = Pipeline([
        ('feature_reduction', feature_reduction_estimator)
    ])
    print(X_imputed)
    print(X_tokenized)
    X_reduced = feature_reduction_pipeline.fit_transform(X_tokenized, iris_df["species"])

    # Model training and hyperparameter tuning
    model_pipeline_steps = []
    param_grids = {}



    if model_type == "regression":
        
        #Random Forest Regressor
        model_pipeline_steps.append(('random_forest', RandomForestRegressor()))
        param_grids['random_forest__n_estimators'] = instructions['design_state_data']['algorithms']['RandomForestRegressor']['max_trees']
        param_grids['random_forest__criterion'] = instructions['design_state_data']['algorithms']['RandomForestRegressor']['feature_sampling_statergy']
        param_grids['random_forest__max_depth'] = instructions['design_state_data']['algorithms']['RandomForestRegressor']['max_depth']
        param_grids['random_forest__min_samples_split'] = instructions['design_state_data']['algorithms']['RandomForestRegressor']['min_depth']
        param_grids['random_forest__min_samples_leaf'] = instructions['design_state_data']['algorithms']['RandomForestRegressor']['min_samples_per_leaf_min_value']
        param_grids['random_forest__n_jobs'] = instructions['design_state_data']['algorithms']['RandomForestRegressor']['parallelism']
        

        #Linear Regression
        model_pipeline_steps.append(('linear_regression', LinearRegression()))
        param_grids['linear_regression__n_jobs'] = instructions['design_state_data']['algorithms']['LinearRegression']['parallelism']
        
        #Ridge Regression
        model_pipeline_steps.append(('ridge_regression', Ridge()))
        param_grids['ridge_regression__alpha'] = instructions['design_state_data']['algorithms']['RidgeRegression']['max_regparam']
        
        
        #Lasso Regression
        model_pipeline_steps.append(('lasso_regression', Lasso()))
        param_grids['lasso_regression__alpha'] = instructions['design_state_data']['algorithms']['LassoRegression']['max_regparam']
        
        
        #ElasticNet Regression
        model_pipeline_steps.append(('elastic_net_regression', ElasticNet()))
        param_grids['elastic_net_regression__alpha'] = instructions['design_state_data']['algorithms']['ElasticNetRegression']['max_regparam']
        param_grids['elastic_net_regression__l1_ratio'] = instructions['design_state_data']['algorithms']['RidgeRegression']['max_elasticnet']

        
        #GBT Regressor
        model_pipeline_steps.append(('gbt_regression', GradientBoostingRegressor()))
        param_grids['gbt_regression__n_estimators'] = instructions['design_state_data']['algorithms']['GBTRegressor']['num_of_BoostingStages']
        param_grids['gbt_regression__learning_rate'] = [instructions['design_state_data']['algorithms']['GBTRegressor']['min_stepsize'], instructions['design_state_data']['algorithms']['GBTRegressor']['max_stepsize']]
        param_grids['gbt_regression__max_depth'] = instructions['design_state_data']['algorithms']['GBTRegressor']['max_depth']
        
    else:
        raise ValueError("No other type of modelling is suppported except regression.")

    model_pipeline = Pipeline(steps=model_pipeline_steps)
    if instructions['design_state_data']['hyperparameters']['stratergy'] == "Grid Search":
        n_jobs = instructions['design_state_data']['hyperparameters']['parallelism']
        scoring = instructions['design_state_data']['hyperparameters']['cross_validation_stratergy']
        grid_search = GridSearchCV(model_pipeline, param_grids, cv=5, n_jobs=n_jobs, scoring=scoring)
        grid_search.fit(X_reduced, iris_df[target_feature])
        
        print("Evaluation Metrics:")
        print(f"Best Estimator: {grid_search.best_estimator_}")
        print(f"Best Score: {grid_search.best_score_}")
        print(f"Best Parameters: {grid_search.best_params_}")
    
    else:
        print("No other method for hyperparameter tunning is currently supported.")
     
    return grid_search.best_estimator_
