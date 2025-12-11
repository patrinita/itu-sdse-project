from mlflow.tracking import MlflowClient

#production model
def get_production_model(model_name):#added
    client = MlflowClient()
    prod_model = [model for model in client.search_model_versions(f"name='{model_name}'") if dict(model)['current_stage']=='Production']
    return prod_model#changed

def evaluate_production_model(prod_model, model_name):#added
    prod_model_exists = len(prod_model)>0

    if prod_model_exists:
        prod_model_version = dict(prod_model[0])['version']
        prod_model_run_id = dict(prod_model[0])['run_id']
        
        print('Production model name: ', model_name)
        print('Production model version:', prod_model_version)
        print('Production model run id:', prod_model_run_id)
    
    else:
        print('No model in production')

def main():#added all lines below
    prod_model = get_production_model(model_name)
    evaluate_production_model(prod_model, model_name)


if __name__ == "__main__":
    main()