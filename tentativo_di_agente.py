import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random as rnd
from collections import deque
import tentativo_di_ambiente as Env
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt 
import os 
os.chdir("C:/Users/macri/Desktop/RICERCA APERTA/DDPG_OPTIMAL_EXECUTION/")

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
rnd.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class NN(nn.Module):
    
    def __init__(self, n_in, n_out, nNodes, nLayers,
                 activation='silu', out_activation=None,
                 scale=1):
        super(NN, self).__init__()
        activation_map = {
            'silu': nn.SiLU(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()
        }
        layers = [nn.Linear(n_in, nNodes), activation_map.get(activation, nn.SiLU())]
        for _ in range(nLayers - 1):
            layers.append(nn.Linear(nNodes, nNodes))
            layers.append(activation_map.get(activation, nn.SiLU()))
        self.hidden_layers = nn.Sequential(*layers)
        self.prop_h_to_out = nn.Linear(nNodes, n_out)
        self.out_activation = out_activation
        self.scale = scale

    def forward(self, x):
        h = self.hidden_layers(x)
        y = self.prop_h_to_out(h)
        if self.out_activation == 'tanh': y = torch.tanh(y)
        elif self.out_activation == 'sigmoid': y = torch.sigmoid(y)
        elif self.out_activation == 'relu': y = torch.relu(y)
        elif self.out_activation == 'leaky_relu': y = nn.functional.leaky_relu(y, negative_slope=0.01)
        elif self.out_activation is None: pass
        else: raise ValueError(f"Unknown output activation: {self.out_activation}")
        return y

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = rnd.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class OUNoise():
    def __init__(self, mu=0.0, theta=0.15, sigma=0.2):
        self.mu, self.theta, self.sigma = mu, theta, sigma
        self.state = 0.0
    def reset(self): self.state = 0.0
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn()
        self.state += dx
        return self.state

class Agent():

    def __init__(
            self, 
            nNodes, nLayers, 
            gamma, lr_Q, lr_pi,
            tau, sched_step_size, 
            environment = Env.Environment(),
            batch_size=64, memory_capacity=1000000, device='cuda' if torch.cuda.is_available() else 'cpu'
            ):
        
        self.env = environment
        self.gamma = gamma
        self.lr_Q = lr_Q
        self.lr_pi = lr_pi
        self.n_nodes = nNodes
        self.n_layers = nLayers
        self.tau = tau
        self.sched_step_size = sched_step_size
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_capacity)
        self.device = device

        self.avg_action_per_episode_history = []
        self.action_trajectory_history = []

        self.__initialize_NNs__()

        self.pi_loss_history = []
        self.Q_loss_history = []
        self.avg_actions_history = []
        self.total_rewards_history = []
        self.action_trajectory_history = []
        self.Q_loss_history = []
        self.pi_loss_history = []

    def __initialize_NNs__(self):
        self.pi = {'net': NN(n_in=3, n_out=1, nNodes=self.n_nodes, 
                                     nLayers=self.n_layers, activation='silu',
                                     out_activation='tanh', scale= self.env.q_0/self.env.N).to(self.device)}
        self.pi['optimizer'], self.pi['scheduler'] = self.__get_optim_sched__(self.pi['net'], self.lr_pi)
        self.pi_target = {'net': copy.deepcopy(self.pi['net'])}
        self.Q_main = {'net' : NN(n_in=4, n_out=1, nNodes=self.n_nodes, 
                                          nLayers=self.n_layers).to(self.device)}
        self.Q_main['optimizer'], self.Q_main['scheduler'] = self.__get_optim_sched__(self.Q_main['net'], self.lr_Q)
        self.Q_target = {'net': copy.deepcopy(self.Q_main['net'])}

    def __get_optim_sched__(self, net, lr):
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.sched_step_size, gamma=0.99)
        return optimizer, scheduler
    
    def plot_training_summary(self, current_episode):
        """
        Crea e salva un grafico riassuntivo dell'addestramento in una griglia 2x2:
        - Rewards per episodio
        - Traiettoria delle azioni (ultimo episodio)
        - Q Loss
        - Policy Loss
        """
        if not self.total_rewards_history:
            return

        print(f"\nGenerazione del riepilogo grafico per l'episodio {current_episode}...")

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Riepilogo Addestramento - Episodio {current_episode}', fontsize=20, y=1.02)

        # --- 1. Rewards per Episodio ---
        ax1 = axes[0, 0]
        ax1.plot(self.total_rewards_history, color='tab:blue')
        ax1.set_title('Rewards per Episodio', fontsize=16)
        ax1.set_xlabel('Episodio', fontsize=12)
        ax1.set_ylabel('Reward Cumulativa', fontsize=12)
        ax1.grid(True)

        # --- 2. Traiettoria delle Azioni (ultimo episodio) ---
        ax2 = axes[0, 1]
        # Ricostruisci la traiettoria dell'inventario dall'ultima traiettoria delle azioni
        last_actions = self.action_trajectory_history[-1]
        inventory_agent = [self.env.q_0]
        for action in last_actions:
            if not np.isnan(action): # Ignora i valori NaN di padding
                inventory_agent.append(inventory_agent[-1] - action)
        
        inventory_twap = np.linspace(self.env.q_0, 0, self.env.N+1)

        ax2.plot(inventory_agent, marker='o', linestyle='-', color='tab:blue', label='Traiettoria Agente')
        ax2.plot(inventory_twap, linestyle='--', color='orange', label='TWAP (Benchmark)')
        ax2.set_title(f'Traiettoria Inventario (Episodio {current_episode})', fontsize=16)
        ax2.set_xlabel('Passo Temporale', fontsize=12)
        ax2.set_ylabel('Inventario Rimanente', fontsize=12)
        ax2.grid(True)
        ax2.legend()
        # --- 3. Q Loss ---
        ax3 = axes[1, 0]
        ax3.plot(self.Q_loss_history, label='Q-Loss', color='tab:blue', alpha=0.7)
        ax3.set_title('Q Loss', fontsize=16)
        ax3.set_xlabel('Passo di Addestramento', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.grid(True)
        ax3.legend()
        ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # --- 4. Policy Loss ---
        ax4 = axes[1, 1]
        ax4.plot(self.pi_loss_history, label='Policy Loss', color='tab:orange', alpha=0.7)
        ax4.set_title('Policy Loss', fontsize=16)
        ax4.set_xlabel('Passo di Addestramento', fontsize=12)
        ax4.set_ylabel('Loss', fontsize=12)
        ax4.grid(True)
        ax4.legend()
        ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # --- Salvataggio del grafico ---
        if not os.path.exists('grafici_training'):
            os.makedirs('grafici_training')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'grafici_training/summary_episode_{current_episode}.png')
        plt.show()
        plt.close(fig)

    def plot_average_action_vs_timestep(self):
        """
        Crea un grafico dell'azione media per ogni timestep,
        calcolata su tutti gli episodi.
        """
        if not self.action_trajectory_history:
            print("Nessuna cronologia delle azioni disponibile per il plotting.")
            return

        print("\nCreazione del grafico: Azione Media vs. Time Step...")

        # Converte la lista di traiettorie in un array NumPy
        action_array = np.array(self.action_trajectory_history)

        # Calcola la media lungo le colonne (axis=0), ignorando i valori NaN
        # Questo dà l'azione media per ogni timestep
        mean_actions_over_time = np.nanmean(action_array, axis=0)

        # Crea l'asse x per i timestep
        timesteps = np.arange(self.env.N)

        # Crea il grafico
        plt.figure(figsize=(12, 6))
        plt.plot(timesteps, mean_actions_over_time, marker='o', linestyle='-', label='Strategia Media dell\'Agente')
        plt.title('Azione Media per Time Step (Strategia di Esecuzione)')
        plt.xlabel('Time Step (t)')
        plt.ylabel('Azione Media (v_t)')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def soft_update(self):
        for target_param, main_param in zip(self.Q_target['net'].parameters(), self.Q_main['net'].parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)
        for target_param, main_param in zip(self.pi_target['net'].parameters(), self.pi['net'].parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

    def learn(self, n_epochs):
        if len(self.memory) < self.batch_size:
            return
        
        for _ in range(n_epochs):
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            # --- Aggiornamento del Critico ---
            with torch.no_grad():
                norm_next_states = self.env.normalize_state(next_states)
                next_actions = self.pi_target['net'](norm_next_states)
                next_Q_values = self.Q_target['net'](torch.cat((norm_next_states, next_actions), dim=1))
                target_Q_values = rewards + (1 - dones.unsqueeze(1)) * self.gamma * next_Q_values
            
            norm_states = self.env.normalize_state(states)
            current_Q_values = self.Q_main['net'](torch.cat((norm_states, actions), dim=1))
            Q_loss = F.mse_loss(current_Q_values, target_Q_values)

            self.Q_main['optimizer'].zero_grad()
            Q_loss.backward()
            self.Q_main['optimizer'].step()
            self.Q_main['scheduler'].step()
            self.Q_loss_history.append(Q_loss.item())
            
            # --- Aggiornamento dell'Attore ---
            for p in self.Q_main['net'].parameters():
                p.requires_grad = False
            
            
            pi_actions = self.pi['net'](norm_states)
            pi_loss = -torch.mean(self.Q_main['net'](torch.cat((norm_states, pi_actions), dim=1)))
            
            self.pi['optimizer'].zero_grad()
            pi_loss.backward()
            self.pi['optimizer'].step()
            self.pi['scheduler'].step()
            self.pi_loss_history.append(pi_loss.item())
            
            for p in self.Q_main['net'].parameters():
                p.requires_grad = True

    def train(self, n_episodes=1000, n_epochs = 1, n_plot=100):
        epsilon = 1.0
        epsilon_decay = 1.0 / n_episodes
        if self.env.lambd > 0.0:
            epsilon_min = 0.1 
        else:
            epsilon_min = 0.1 # necessita rumore in meno forse per twap?

        for episode in tqdm(range(n_episodes)):
            state = self.env.reset()
            total_reward = 0
            
            episode_actions = []
            curr_price = []

            # Il ciclo for garantisce la terminazione, il break gestisce la fine anticipata
            for t_step in range(self.env.N):

                if t_step == self.env.N-1:
                    action = state[:, 1].clone().detach()  # Ultimo passo, non si può agire

                norm_state = self.env.normalize_state(state)
                with torch.no_grad():
                    # Aggiungi rumore gaussiano per l'esplorazione
                    if self.env.lambd > 0.0:
                        max_action = 20 * self.env.q_0 / self.env.N # Limite superiore ragionevole
                    else:
                        max_action = 2 * self.env.q_0 / self.env.N
                    # Azione da rete [-1, 1]
                    base_action = self.pi['net'](norm_state.squeeze(0).to(self.device))

                    # Aggiungi rumore e clippa
                    noise = torch.randn(1, device=self.device) * (epsilon * max_action)


                    base_action = base_action * self.env.q_0 / self.env.N  + noise
                    #action_with_noise = base_action + noise
                    #clipped_action = torch.clamp(action_with_noise, -1.0, 1.0)

                    # Scala l'azione per essere sempre positiva [0, max_action]
                    #action = (clipped_action + 1) / 2.0 * max_action

                    #action = clipped_action * max_action  # Scala l'azione per essere tra 0 e max_action

                    action = torch.clamp(base_action, torch.zeros_like(base_action), state[:,1])
                      # Assicura che l'azione sia tra 0 e max_action

                next_state, reward, done, _ = self.env.step(state, action)
                
                episode_actions.append(action.item())

                reward 
                
                # Usa .squeeze(0) per rimuovere la dimensione batch prima di salvare
                self.memory.push(
                    state.squeeze(0).to(self.device),
                    action.to(self.device), # L'azione è già 1D
                    reward.to(self.device),
                    next_state.squeeze(0).to(self.device),
                    done
                )
                
                state = next_state
                total_reward += reward.item() 
                price = state[:, 0].item()  # Prezzo corrente
                curr_price.append(price)
            
            # Calcola e salva il prezzo medio di questo episodio
            curr_price.append(state[:, 0].item())  # Aggiungi l'ultimo prezzo

                #if done:
                #    break # Interrompi il ciclo se l'episodio è finito
            
            self.total_rewards_history.append(total_reward)
            padding = self.env.N - len(episode_actions)
            self.action_trajectory_history.append(episode_actions + [np.nan] * padding)

            if len(self.memory) >= self.batch_size: 
                self.learn(n_epochs)

            if episode_actions: # Evita errori se l'episodio non ha avuto azioni
                mean_action_this_episode = np.mean(episode_actions)
                self.avg_actions_history.append(mean_action_this_episode)

            padding_needed = self.env.N - len(episode_actions)
            padded_actions = episode_actions + [np.nan] * padding_needed
            self.action_trajectory_history.append(padded_actions)

            # Stampa un riepilogo dell'episodio
            if episode % 100 == 0:
                print(f"\nEpisodio {episode + 1}/{n_episodes} | Ricompensa Totale: {total_reward:.2f} | Azione Media: {mean_action_this_episode:.2f} | Epsilon: {epsilon:.2f}")
                avg_price_history = curr_price
                # plt.plot(np.arange(self.env.N+1), avg_price_history, label='Prezzo Medio per Episodio', color='tab:green')
                # plt.show()
            self.soft_update()
            
            # Decadimento di epsilon
            epsilon = max(epsilon_min, epsilon - epsilon_decay)
            if (episode + 1) % n_plot == 0:
                self.plot_training_summary(current_episode=episode + 1)

#%%
if __name__ == "__main__":

    ambiente = Env.Environment(
        S_0=20, q_0=40, kappa=0.25, alpha=0.05, sigma=0.1, mu=0.0, 
        dt=0.1, T=2, lambd = 0.0, theta = 0.0, ##################################
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    agent = Agent(
        nNodes=64, nLayers=3,
        gamma=0.99, lr_Q=0.1, lr_pi=0.1,
        tau=0.001, sched_step_size=1_000,
        environment=ambiente,
        batch_size=128, memory_capacity=100000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    import os
    print(f"La cartella di lavoro corrente è: {os.getcwd()}")
    # Esegui l'addestramento
    agent.train(n_episodes=5_000, n_epochs=10, n_plot=100)

    print("Addestramento completato. Generazione del grafico finale...")
    agent.plot_training_summary(current_episode=5000)
    print('fatto!')
#%%
# Salva i pesi dei modelli Q_main e pi
torch.save(agent.Q_main['net'].state_dict(), 'Q_main_weights_risk_neutral.pth')
torch.save(agent.pi['net'].state_dict(), 'pi_weights_risk_neutral.pth')
print("Salvataggio dei pesi completato.")
# %%
