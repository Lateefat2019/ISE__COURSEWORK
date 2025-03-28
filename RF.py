import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_rel


def main():
    """
    Parameters:
    systems (list): List of systems containing CSV datasets.
    num_repeats (int): Number of times to repeat the evaluation for avoiding stochastic bias.
    train_frac (float): Fraction of data to use for training.
    random_seed (int): Initial random seed to ensure the results are reproducible
    """

    # Specify the parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 3
    train_frac = 0.7
    random_seed = 1

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=random_seed),
    }

    parameter = {
        'Random Forest': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 5]
        }
    }

    all_results = []

    for current_system in systems:
        datasets_location = f'datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f'\n> System: {current_system}, Dataset: {csv_file}, Training data fraction: {train_frac}, Number of repeats: {num_repeats}')
            data = pd.read_csv(os.path.join(datasets_location, csv_file))
            metrics = {model_name: {'MAPE': [], 'MAE': [], 'RMSE': []} for model_name in models}

            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                for model_name, model in models.items():

                    if model_name in parameter:
                        grid_search = RandomizedSearchCV(
                            model,
                            param_distributions=parameter[model_name],
                            n_iter=5,
                            cv=3,
                            scoring='neg_mean_absolute_error',
                            n_jobs=-1,
                            verbose=0,
                            random_state=1
                        )
                        grid_search.fit(training_X, training_Y)
                        model = grid_search.best_estimator_
                        print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
                    else:
                        model.fit(training_X, training_Y)

                    predictions = model.predict(testing_X)

                    mape = mean_absolute_percentage_error(testing_Y, predictions)
                    mae = mean_absolute_error(testing_Y, predictions)
                    rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                    metrics[model_name]['MAPE'].append(mape)
                    metrics[model_name]['MAE'].append(mae)
                    metrics[model_name]['RMSE'].append(rmse)

            print(f"\nResults for {csv_file}:")
            for model_name, model_metrics in metrics.items():
                avg_mape = np.mean(model_metrics['MAPE'])
                avg_mae = np.mean(model_metrics['MAE'])
                avg_rmse = np.mean(model_metrics['RMSE'])

                print(f"Model: {model_name} : Average MAPE: {avg_mape:.2f}, Average MAE: {avg_mae:.2f}, Average RMSE: {avg_rmse:.2f}")

                all_results.append({
                    'System': current_system,
                    'Dataset': csv_file,
                    'Model': model_name,
                    'Avg_MAPE': round(avg_mape, 4),
                    'Avg_MAE': round(avg_mae, 4),
                    'Avg_RMSE': round(avg_rmse, 4)
                })

    # Save all results to Excel
    results_df = pd.DataFrame(all_results)
    output_path = 'randomforest_results.xlsx'
    results_df.to_excel(output_path, index=False)
    print(f"\n✅ All results saved to Excel file: {output_path}")

    # Perform paired t-test (based on MAE)
    rf_mae_values = []
    lr_mae_values = []

    grouped = results_df.groupby(['System', 'Dataset'])

    for (_, dataset_group) in grouped:
        rf_row = dataset_group[dataset_group['Model'] == 'Random Forest']
        lr_row = dataset_group[dataset_group['Model'] == 'Linear Regression']
        if not rf_row.empty and not lr_row.empty:
            rf_mae_values.append(rf_row['Avg_MAE'].values[0])
            lr_mae_values.append(lr_row['Avg_MAE'].values[0])

    rf_mae_values = np.array(rf_mae_values)
    lr_mae_values = np.array(lr_mae_values)

    if len(rf_mae_values) == len(lr_mae_values) and len(rf_mae_values) > 1:
        t_stat, p_value = ttest_rel(lr_mae_values, rf_mae_values)
        print("\n🔍 Paired t-test between Linear Regression and Random Forest (based on MAE):")
        print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("✅ The difference is statistically significant (p < 0.05). Random Forest performs better.")
        else:
            print("❌ No statistically significant difference (p ≥ 0.05).")
    else:
        print("\n⚠️ Not enough paired results to perform t-test.")

if __name__ == "__main__":
    main()
