# Import the function `entrenar_modelo_mnist` from the `neuronal` module in the `src` package
from src.neuronal import entrenar_modelo_mnist

def main():
    """
    Main entry point of the program.
    """
    # Print a message indicating the start of the MNIST model training
    print("Iniciando el entrenamiento del modelo MNIST...")
    
    # Call the function to train the MNIST model
    entrenar_modelo_mnist()
    
    # Print a message indicating the completion of the training
    print("Entrenamiento completado.")

# Check if the script is being run directly (not imported as a module)
if __name__ == "__main__":
    # Call the main function to start the program
    main()
