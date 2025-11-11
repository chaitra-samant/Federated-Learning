from mpi4py import MPI
import torch
from federated.client import FederatedClient
from federated.server import FederatedServer
from federated.fed_avg import fed_avg

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

NUM_ROUNDS = 5
EPOCHS = 1

def main():
    if RANK == 0:
        # Server process
        server = FederatedServer()
        global_model_params = None

        for round_idx in range(NUM_ROUNDS):
            print(f"Server: Starting round {round_idx+1}/{NUM_ROUNDS}")

            # Send global parameters to clients
            for client_rank in range(1, SIZE):
                COMM.send(global_model_params, dest=client_rank, tag=11)

            # Receive updated params and sizes from clients
            client_params_list = []
            client_sizes = []
            for client_rank in range(1, SIZE):
                client_data = COMM.recv(source=client_rank, tag=22)
                client_params_list.append(client_data['params'])
                client_sizes.append(client_data['size'])

            # Aggregate using FedAvg
            global_model_params = fed_avg(client_params_list, client_sizes)
            print(f"Server: Completed aggregation for round {round_idx+1}")

        print("Training Complete")

    else:
        # Client process
        client_id = RANK
        client = FederatedClient(client_id)

        # Block waiting for initial params (None at first)
        global_params = COMM.recv(source=0, tag=11)
        if global_params:
            client.set_parameters(global_params)

        for round_idx in range(NUM_ROUNDS):
            print(f"Client {client_id}: Training round {round_idx+1}")

            client.set_parameters(global_params) if global_params else None

            client.train(epochs=EPOCHS)
            acc = client.evaluate()

            # Send updated params and sample size back to server
            params = client.get_parameters()
            data = {'params': params, 'size': len(client.train_loader.dataset)}
            COMM.send(data, dest=0, tag=22)

            # Wait for updated global params
            global_params = COMM.recv(source=0, tag=11)

if __name__ == "__main__":
    main()