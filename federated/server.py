class FederatedServer:
    def __init__(self):
        self.global_parameters = None

    def set_global_parameters(self, parameters):
        self.global_parameters = parameters

    def aggregate(self, client_params_list, client_sizes):
        """Perform weighted FedAvg."""
        aggregated_params = {}
        total_size = sum(client_sizes)
        for k in client_params_list[0].keys():
            aggregated_params[k] = sum(client_params[k].float() * client_sizes[i] / total_size 
                                     for i, client_params in enumerate(client_params_list))
        self.global_parameters = aggregated_params
        return self.global_parameters

    def get_global_parameters(self):
        return self.global_parameters