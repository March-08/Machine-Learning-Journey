import numpy as np
import tensorflow as tf

from utils import test_generator

batch_size = 64
path_to_test = "./data/test"

model =  tf.keras.models.load_model("./Models")
test_set = test_generator(batch_size = batch_size, path_to_test = path_to_test)

preds = model.predict(test_set).flatten()

output = pd.DataFrame({'id': np.arange(1,12501), 'label': preds})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


