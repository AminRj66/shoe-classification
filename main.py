import time
import tensorflow as tf
from src import Utility, PrepareData, FineTunedMobileNet, FineTunedEfficientNet, SeparatedClassifier

if __name__ == "__main__":
    start_time = time.time()

    util = Utility()

    # initiate objects
    params = util.load_params("config/config.yml")
    data_object = PrepareData(config_dir="config", utility=util)

    # set device option
    util.set_device_option(params['device'])

    with tf.device(params['device']):

        # begin experiment
        if params['model_type'] == "fine_tuned_mobile_net":
            model_object = FineTunedMobileNet(config_dir="config", n_classes=data_object.n_classes, utility=util)
            model_object.run_fine_tuned_mobile_net(data_object)

        elif params['model_type'] == "fine_tuned_EfficientNet":
            model_object = FineTunedEfficientNet(config_dir="config", n_classes=data_object.n_classes, utility=util)
            model_object.run_fine_tuned_vit(data_object)

        elif params['model_type'] == "separated_classifier":
            model_object = SeparatedClassifier(config_dir="config", n_classes=data_object.n_classes, utility=util)
            model_object.run_separated_classifier(data_object)

        print("--- %s seconds ---" % (time.time() - start_time))
