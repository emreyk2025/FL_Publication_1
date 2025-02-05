import flwr as fl

class FlowerClient(fl.client.NumPyClient):
    
    # " -> None" is a type hint, informing the reader to expect None out of this init method.
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, model_func) -> None:
        
        self.model = model_func
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        # Update local model parameters
        self.model.set_weights(parameters)


        epochs: int = config.get("epochs", 10)
        batch_size: int = config.get("batch_size", 32)

        # Model training
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size = batch_size,
            epochs = epochs,
            validation_data=(self.x_val,self.y_val),         
            verbose=0                                        
        )                                                    

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][-1],
            "mae": history.history["mean_absolute_error"][-1]
        }
        return parameters_prime, num_examples_train, results
    
    def evaluate(self, parameters, config):

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data and return results
        loss, mae = self.model.evaluate(self.x_test, self.y_test, 32)
        num_examples_test = len(self.x_test)

        metrics = {
            "mae": float(mae)
        }

        return float(loss), num_examples_test, metrics
    

'''
NumPyClient is only one kind of client available in Flower library, there are others as well. In fact
Flower doesnt actually use the FlowerNumPyClient object directly. Instead, it wraps the object to 
make it look like a subclass of flwr.client.Client, not flwr.client.NumPyClient.
NumPyClient is just a convenience abstraction built on top of Client.
The difference between Client and NumPyClient comes down to parameter serialization (Turning 
parameters to bytes while sending them to nodes) and deserialization.
NumpyClient does this (un)serialization automatically. In Client you should do it manually.
NumpyClient makes our work a breeze with ML models that have good Numpy support.
'''




