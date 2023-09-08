#Carga de bibliotecas
import tensorflow as tf
import datetime

mnist = tf.keras.datasets.mnist #carga de los datos
(x_train, y_train),(x_test, y_test) = mnist.load_data() #Split data
x_train, x_test = x_train / 255.0, x_test / 255.0 #normalización de los datos

#Creamos la arquitectura de un modelo de clasificación
def create_model(dropout,layers_dense):
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='layers_flatten'),
    tf.keras.layers.Dense(512, activation='relu', name='layers_dense'),
    tf.keras.layers.Dropout(dropout, name='layers_dropout'),
    tf.keras.layers.Dense(layers_dense, activation='softmax', name='layers_dense_2')
  ])

def training(epochs,dropout,layers_dense,optimizer,filename):
    #instancia del modelo
    model = create_model(dropout,layers_dense)
    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    log_dir = "logs/fit/"+str(filename) +"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S") #Ubicación para almacenar logs
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #Entrenamiento del modelo
    model.fit(x=x_train, y=y_train, 
            epochs=epochs, validation_data=(x_test, y_test), 
            callbacks=[tensorboard_callback]) #nos aseguramos que los logs son creados y almacenados

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--layers_dense", type=int)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()

    print("\n\n*** Starting training ***")
    from knockknock import discord_sender
    webhook_url = "https://discord.com/api/webhooks/1149116324417654936/E9jIEvnfvY_8vR5IHHPMcLdEoGI-8q4TDerNAL6Fj8usuO7yDfLHg0LNnsegxrnvvwAn"
    @discord_sender(webhook_url=webhook_url)
    def train_model_email_notify(epochs,dropout,layers_dense,optimizer,filename):
      return training(epochs,dropout,layers_dense,optimizer,filename)

    train_model_email_notify(args.epochs,args.dropout,args.layers_dense,args.optimizer,args.filename)