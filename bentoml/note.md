
## Deploying Your Prediction Service

## Links
* [Building Bentos Documentation](https://docs.bentoml.org/en/latest/concepts/bento.html)

## Notes

In this section we are going to look at BentoML cli and what operations BentoML is performing behind the scenes.

We can get a list of saved model in the terminal using the commmand `bentoml models list`. This command shows all the saved models and their tags, module, size, and the time they were created at. For instance:

```bash
 Tag                           Module           Size        Creation Time
 credit_risk_model:l652ugcqk…  bentoml.xgboost  197.77 KiB  2022-10-20 08:29:54
```

We can use `bentoml models list -o json|yaml|table` to display the output in one of the given format.

Running the command `bentoml models get credit_risk_model:l652ugcqkgefhd7k` displays the information about the model which looks like:

```yaml
name: credit_risk_model
version: l652ugcqkgefhd7k
module: bentoml.xgboost
labels: {}
options:
  model_class: Booster
metadata: {}
context:
  framework_name: xgboost
  framework_versions:
    xgboost: 1.6.2
  bentoml_version: 1.0.7
  python_version: 3.10.6
signatures:
  predict:
    batchable: false
api_version: v2
creation_time: '2022-10-20T08:29:54.706593+00:00'
```

Important thing to note here is that the version of the XGBoost in the `framework_versions` has to be same as the model was trained with otherwise we might get inconsistent results. The BentoML pulls these dependencies automatically and generates this file for convenience.

The next we want to do is, creating the file `bentofile.yaml`:

```yaml
service: "service.py:svc" # Specify entrypoint and service name
labels: # Labels related to the project for reminder (the provided labels are just for example)
  owner: bentoml-team
  project: gallery
include:
- "*.py" # A pattern for matching which files to include in the bento build
python:
  packages: # Additional pip packages required by the service
    - xgboost
    - sklearn
```

Once we have our `service.py` and `bentofile.yaml` files ready we can build the bento by running the command `bentoml build`. It will look in the service.py file to get all models being used and into bentofile.yaml file to get all the dependencies and creates one single deployable directory for us. The output will look something like this:

```bash
Successfully built Bento(tag="credit_risk_classifier:kdelkqsqms4i2b6d")
```

We can look into this directory by locating `cd ~/bentoml/bentos/credit_risk_classifier/kdelkqsqms4i2b6d/` and the file structure may look like this:

```bash
.
├── README.md # readme file
├── apis
│   └── openapi.yaml # openapi file to enable Swagger UI
├── bento.yaml # bento file to bind everything together
├── env # environment related directory
│   ├── docker # auto generate dockerfile (also can be customized)
│   │   ├── Dockerfile
│   │   └── entrypoint.sh
│   └── python # requirments for installation
│       ├── install.sh
│       ├── requirements.txt
│       └── version.txt
├── models # trained model(s)
│   └── credit_risk_model
│       ├── l652ugcqkgefhd7k
│       │   ├── custom_objects.pkl # custom objects (in our case DictVectorizer)
│       │   ├── model.yaml # model metadate
│       │   └── saved_model.ubj # saved model
│       └── latest
└── src
    └── service.py # bentoml service file for endpoint
```

The idea behind the structure like this is to provide standardized way that a machine learning service might required.

Now the last thing we need to do is to build the docker image. This can be done with `bentoml containerize credit_risk_classifier:kdelkqsqms4i2b6d`.

> Note: We need to have Docker installed before running this command.

Once the docker image is built successfully, we can run `docker run -it --rm -p 3000:3000 containerize credit_risk_classifier:kdelkqsqms4i2b6d` to see if everything is working as expected. We are exposing 3000 port to map with the service port which is also 3000 and this should take us to Swagger UI page again.

## Sending, Receiving and Validating Data

## Instructor Clarifications
**Please remember pydantic is not included by BentoML**
It must be installed using ```pip install pydantic```

Also remember to include it in your bentofile.yaml as a dependency when building your bento

## Notes

Data validation is another great feature on BentoML that ensures the data transferation is valid and reliable. We can integrate Python library Pydatic with BentoML for this purpose.

Pydantic can be installed with `pip install pydantic`, after that we need to import the `BaseModel` class from the library and create our custom class for data validation:

```python
# Create pydantic base class to create data schema for validation
class CreditApplication(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int
```

Our model is trained on 13 features of different data types and the BaseModel will ensure that we are always recieving them for the model prediction.

Next we need to implement pass the class in our bentoml service:

```python
# Pass pydantic class in the application
@svc.api(input=JSON(pydantic_model=CreditApplication), output=JSON()) # decorate endpoint as in json format for input and output
def classify(credit_application):
    # transform pydantic class to dict to extract key-value pairs 
    application = credit_application.dict()
    # transform data from client using dictvectorizer
    vector = dv.transform(application)
    # make predictions using 'runner.predict.run(input)' instead of 'model.predict'
    prediction = model_runner.predict.run(vector) 
```

Along the `JSON()`, BentoML uses various other descriptors in the input and output specification of the service api, for example, NumpyNdarray(), PandasDataFrame(), Text(), and many more.