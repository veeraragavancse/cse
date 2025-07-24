import tkinter as tk
import numpy as np
import random

# Genetic Algorithm Parameters
population_size = 20  # Increase population size for better convergence
num_generations = 50  # Increase the number of generations
crossover_rate = 0.8
mutation_rate = 0.01
entries = []

# Set a random seed for consistency
random.seed(42)
np.random.seed(42)

# Student Data (Replace with your actual data)
previous_marks = [[85, 90, 78],
                  [72, 82, 85],
                  [92, 88, 95],
                  [80, 85, 90]]  # New data entered by the user

# Final marks (target values to predict) for each student
final_marks = [88, 78, 95, 85]  # Replace this with actual final assessments

# Function to generate an initial population of individuals (chromosomes)
def generate_initial_population():
    population = []
    for _ in range(population_size):
        individual = [random.uniform(0, 1) for _ in range(len(previous_marks[0]))]
        population.append(individual)
    return population

# Function to calculate the fitness of an individual
def calculate_fitness(individual, previous_marks, final_marks):
    predicted_marks = np.dot(previous_marks, individual)
    # Fitness is the error between predicted and actual final marks (mean squared error)
    fitness = np.mean((predicted_marks - final_marks) ** 2)
    return fitness

# Function to select parents using tournament selection
def tournament_selection(population, previous_marks, final_marks, tournament_size=3):
    selected_parents = []
    for _ in range(2):
        tournament_candidates = random.sample(population, tournament_size)
        best_individual = min(tournament_candidates, key=lambda x: calculate_fitness(x, previous_marks, final_marks))
        selected_parents.append(best_individual)
    return selected_parents

# Function to perform crossover
def crossover(parent1, parent2):
    child = []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < crossover_rate:
            child.append(gene1)
        else:
            child.append(gene2)
    return child

# Function to perform mutation
def mutation(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(0, 1)
    return individual

# Function to create the Tkinter GUI
def create_gui():
    window = tk.Tk()
    window.title("Student Assessment Predictor")

    # Input labels and entry fields for previous marks
    labels = []
    
    for i in range(len(previous_marks[0])):
        label = tk.Label(window, text=f"Previous Marks {i+1}:")
        entry = tk.Entry(window)
        label.grid(row=i, column=0)
        entry.grid(row=i, column=1)
        labels.append(label)
        entries.append(entry)

    # Button to trigger prediction
    predict_button = tk.Button(window, text="Predict", command=lambda: predict(predicted_assessment_value))
    predict_button.grid(row=len(previous_marks[0]), columnspan=2)

    # Label to display the predicted assessment
    predicted_assessment_label = tk.Label(window, text="Predicted Assessment:")
    predicted_assessment_label.grid(row=len(previous_marks[0]) + 1, columnspan=2)

    # Label to display the predicted assessment value
    predicted_assessment_value = tk.Label(window, text="")
    predicted_assessment_value.grid(row=len(previous_marks[0]) + 2, columnspan=2)

    window.mainloop()

# Function to perform the prediction using the genetic algorithm
def predict(predicted_assessment_label):
    # Retrieve previous marks from the GUI
    previous_marks_input = [float(entry.get()) for entry in entries]

    # Update the previous_marks data with the new input (in place of the last student)
    previous_marks[-1] = previous_marks_input

    # Initialize population
    population = generate_initial_population()

    best_fitness = float("inf")
    best_individual = None

    # Run the genetic algorithm
    for generation in range(num_generations):
        # Calculate fitness for each individual in the population
        fitness_scores = [calculate_fitness(individual, previous_marks, final_marks) for individual in population]
        
        # Select the best individual based on fitness
        current_best_individual = min(population, key=lambda x: calculate_fitness(x, previous_marks, final_marks))
        current_best_fitness = calculate_fitness(current_best_individual, previous_marks, final_marks)

        # If the current best individual is better, update the global best
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual

        # Selection of parents
        parents = [tournament_selection(population, previous_marks, final_marks) for _ in range(population_size // 2)]

        # Generate offspring through crossover
        offspring = [crossover(parent1, parent2) for parent1, parent2 in parents]

        # Apply mutation
        mutated_offspring = [mutation(child) for child in offspring]

        # Replace worst individuals with offspring
        population = sorted(population + mutated_offspring, key=lambda x: calculate_fitness(x, previous_marks, final_marks))[:population_size]

    # Predict the final assessment using the best individual
    predicted_assessment = np.dot(previous_marks_input, best_individual)

    # Update the GUI with the predicted assessment
    predicted_assessment_label.config(text=f"{predicted_assessment:.2f}")

# Create the GUI
create_gui()
