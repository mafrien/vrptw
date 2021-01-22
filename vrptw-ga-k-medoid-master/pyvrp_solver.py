import pandas as pd

from libs.pyVRP import build_coordinates, build_distance_matrix, genetic_algorithm_vrp, plot_tour_coordinates


class PyVRPSolver:
    def __init__(self):
        # Parameters - Model
        self.n_depots = 1  # The first n rows of the distance_matrix or coordinates
        self.time_window = 'with'  # 'with', 'without'
        self.route = 'open'  # 'open', 'closed'
        self.model = 'tsp'  # 'tsp', 'mtsp', 'vrp'
        self.graph = True  # True, False

        # Parameters - Vehicle
        self.vehicle_types = 1  # One Type of Vehicle: A
        self.fixed_cost = [1]  # Fixed Cost for Vehicle A = 30
        self.variable_cost = [1]  # Variable Cost for Vehicle A = 2
        self.capacity = [1000]  # Capacity of Vehicle A = 150
        self.velocity = [
            1]  # Average Velocity of Vehicle A = 70. The Average Velocity Value is Used as a Constant that Divides the Distance Matrix.
        self.fleet_size = []  # An Empty List, e.g  fleet_size = [ ], means that the Fleet is Infinite. Non-Empty List, e.g  fleet_size = [15, 7], means that there are available 15 vehicles of type A and 7 vehicles of type B

        # Parameters - GA
        self.penalty_value = 10000  # GA Target Function Penalty Value for Violating the Problem Constraints
        self.population_size = 50  # GA Population Size
        self.mutation_rate = 0.10  # GA Mutation Rate
        self.elite = 1  # GA Elite Member(s) - Total Number of Best Individual(s) that (is)are Maintained in Each Generation
        self.generations = 10  # GA Number of Generations

    def solve(self, launch_count):
        ga_reports = []
        plots_data = []
        for i in range(launch_count):
            coordinates = pd.read_csv('result/coords{}.txt'.format(i), sep=' ')
            coordinates = coordinates.values
            distance_matrix = build_distance_matrix(coordinates)
            parameters = pd.read_csv('result/params{}.txt'.format(i), sep=' ')
            parameters = parameters.values

            # Call GA Function
            ga_report, ga_vrp = genetic_algorithm_vrp(coordinates, distance_matrix, parameters,
                                                      self.population_size,
                                                      self.route, self.model, self.time_window, self.mutation_rate, self.elite,
                                                      self.generations, self.penalty_value, self.graph)

            plot_data = {'coordinates': coordinates, 'ga_vrp': ga_vrp, 'route': self.route}
            plots_data.append(plot_data)

            # Solution Report
            print(ga_report)

            # Save Solution Report
            ga_report.to_csv('tsptw_result/report{}.csv'.format(i), sep=' ', index=False)

            ga_reports.append(ga_report)

        return ga_reports, plots_data
