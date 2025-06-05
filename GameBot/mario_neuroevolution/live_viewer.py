#!/usr/bin/env python3
"""
Live viewer for watching Mario agents play in real-time.
Provides a simple UI to load and watch trained agents.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import neat
import pickle
import threading
import time
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from agents.mario_agent import MarioAgent


class MarioViewer:
    """GUI for watching Mario agents play."""
    
    def __init__(self):
        """Initialize the Mario viewer GUI."""
        self.root = tk.Tk()
        self.root.title("Mario AI Viewer")
        self.root.geometry("600x400")
        
        # Variables
        self.genome = None
        self.config = None
        self.is_playing = False
        self.play_thread = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Mario AI Viewer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Load genome section
        genome_frame = ttk.LabelFrame(main_frame, text="Load Trained Agent", padding="10")
        genome_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(genome_frame, text="Genome file:").grid(row=0, column=0, sticky=tk.W)
        self.genome_path_var = tk.StringVar()
        ttk.Entry(genome_frame, textvariable=self.genome_path_var, width=50).grid(
            row=0, column=1, padx=5)
        ttk.Button(genome_frame, text="Browse", 
                  command=self.browse_genome).grid(row=0, column=2, padx=5)
        
        ttk.Label(genome_frame, text="Config file:").grid(row=1, column=0, sticky=tk.W, pady=(5,0))
        self.config_path_var = tk.StringVar(value="configs/neat_config.txt")
        ttk.Entry(genome_frame, textvariable=self.config_path_var, width=50).grid(
            row=1, column=1, padx=5, pady=(5,0))
        ttk.Button(genome_frame, text="Browse", 
                  command=self.browse_config).grid(row=1, column=2, padx=5, pady=(5,0))
        
        # Control section
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.play_button = ttk.Button(control_frame, text="Play Agent", 
                                     command=self.toggle_play)
        self.play_button.grid(row=0, column=0, padx=5)
        
        ttk.Button(control_frame, text="Stop", 
                  command=self.stop_play).grid(row=0, column=1, padx=5)
        
        # Settings section
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(settings_frame, text="Episodes to play:").grid(row=0, column=0, sticky=tk.W)
        self.episodes_var = tk.StringVar(value="3")
        ttk.Entry(settings_frame, textvariable=self.episodes_var, width=10).grid(
            row=0, column=1, padx=5)
        
        self.render_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Show game window", 
                       variable=self.render_var).grid(row=1, column=0, sticky=tk.W, pady=5)
        
        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.status_text = tk.Text(status_frame, height=8, width=70)
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        
    def browse_genome(self):
        """Browse for genome file."""
        filename = filedialog.askopenfilename(
            title="Select Genome File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            self.genome_path_var.set(filename)
            
    def browse_config(self):
        """Browse for config file."""
        filename = filedialog.askopenfilename(
            title="Select Config File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            self.config_path_var.set(filename)
            
    def log_status(self, message):
        """Add message to status log."""
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        
    def load_agent(self):
        """Load the genome and config."""
        try:
            # Load config
            config_path = self.config_path_var.get()
            if not config_path:
                messagebox.showerror("Error", "Please select a config file")
                return False
                
            self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    config_path)
            self.log_status(f"Loaded config from {config_path}")
            
            # Load genome
            genome_path = self.genome_path_var.get()
            if not genome_path:
                messagebox.showerror("Error", "Please select a genome file")
                return False
                
            with open(genome_path, 'rb') as f:
                self.genome = pickle.load(f)
            self.log_status(f"Loaded genome from {genome_path}")
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load agent: {str(e)}")
            self.log_status(f"Error loading agent: {str(e)}")
            return False
            
    def play_agent(self):
        """Play the loaded agent."""
        if not self.load_agent():
            return
            
        try:
            # Create neural network
            net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
            
            # Create environment
            env = SuperMarioBrosEnv()
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            
            # Create agent
            agent = MarioAgent(self.genome, self.config, 'simple')
            
            episodes = int(self.episodes_var.get())
            render = self.render_var.get()
            
            self.log_status(f"Starting {episodes} episodes with render={render}")
            
            for episode in range(episodes):
                if not self.is_playing:
                    break
                    
                self.log_status(f"Episode {episode + 1}/{episodes}")
                
                state = env.reset()
                total_reward = 0
                steps = 0
                max_distance = 0
                
                while self.is_playing:
                    if render:
                        env.render()
                    
                    # Get action from neural network
                    processed_state = agent.preprocess_state(state)
                    output = net.activate(processed_state)
                    action = agent.get_action(output)
                    
                    # Take step
                    state, reward, done, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    current_distance = info.get('x_pos', 0)
                    max_distance = max(max_distance, current_distance)
                    
                    if done:
                        break
                        
                    # Add small delay to make it watchable
                    if render:
                        time.sleep(0.03)
                
                self.log_status(f"Episode {episode + 1} finished:")
                self.log_status(f"  Reward: {total_reward:.2f}")
                self.log_status(f"  Distance: {max_distance}")
                self.log_status(f"  Score: {info.get('score', 0)}")
                self.log_status(f"  Steps: {steps}")
                self.log_status("---")
            
            env.close()
            self.log_status("Finished playing all episodes")
            
        except Exception as e:
            self.log_status(f"Error during play: {str(e)}")
        finally:
            self.is_playing = False
            self.play_button.config(text="Play Agent")
            
    def toggle_play(self):
        """Toggle playing state."""
        if not self.is_playing:
            self.is_playing = True
            self.play_button.config(text="Playing...")
            
            # Start playing in separate thread
            self.play_thread = threading.Thread(target=self.play_agent)
            self.play_thread.daemon = True
            self.play_thread.start()
        else:
            self.stop_play()
            
    def stop_play(self):
        """Stop playing."""
        self.is_playing = False
        self.play_button.config(text="Play Agent")
        self.log_status("Stopping playback...")
        
    def run(self):
        """Run the GUI."""
        self.log_status("Mario AI Viewer started")
        self.log_status("1. Load a trained genome file")
        self.log_status("2. Adjust settings if needed")
        self.log_status("3. Click 'Play Agent' to watch")
        self.root.mainloop()


if __name__ == "__main__":
    viewer = MarioViewer()
    viewer.run()