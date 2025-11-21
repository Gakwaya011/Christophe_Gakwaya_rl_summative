# environment/rendering.py

import pygame
import numpy as np

class PygameRenderer:
    """Handles the Pygame visualization for the AthleteRecoveryEnv."""

    def __init__(self, screen_width=800, screen_height=600):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("RL Athlete Recovery Dashboard")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)
        self.is_open = True
        
        # Define colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 200, 0)
        self.RED = (200, 0, 0)
        self.BLUE = (0, 0, 200)

    def render_frame(self, state, action_name, episode_reward, current_step):
        """Draws the current state of the environment."""
        
        if not self.is_open:
            return

        self.screen.fill(self.WHITE)
        
        hrv, fatigue, deficit, intensity, days = state

        # --- Section 1: Metrics Dashboard (Top Left) ---
        text_y = 50
        self._draw_text(f"Day: {int(days)} / 30", 50, text_y, self.BLACK)
        text_y += 30
        self._draw_text(f"Cumulative Reward: {episode_reward:.2f}", 50, text_y, self.BLUE)
        text_y += 30
        self._draw_text(f"Action Recommended: {action_name}", 50, text_y, self.GREEN)

        # --- Section 2: Bar Indicators for Key Metrics ---
        
        # HRV (Recovery) Bar - Target High (Green)
        self._draw_bar_meter(250, 50, 200, 30, "HRV (Recovery)", hrv, 1.0, self.GREEN, True)
        
        # Fatigue Score Bar - Target Low (Red)
        self._draw_bar_meter(250, 100, 200, 30, "Fatigue Score", fatigue, 5.0, self.RED, False)
        
        # Workout Intensity Bar - Target Medium/High (Blue)
        self._draw_bar_meter(250, 150, 200, 30, "Next Day Intensity", intensity, 1.0, self.BLUE, True)
        
        # --- Section 3: Performance Pathway (Metaphorical Grid) ---
        # Visualize the agent's progress towards optimal performance
        self._draw_performance_pathway(self.screen_width - 250, 50, hrv, fatigue)

        pygame.display.flip()
        self.clock.tick(15) # Limit frame rate

    def _draw_text(self, text, x, y, color):
        surface = self.font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def _draw_bar_meter(self, x, y, width, height, label, value, max_value, color, is_positive):
        """Draws a horizontal bar meter."""
        
        # Text label
        self._draw_text(label, x, y - 20, self.BLACK)
        
        # Bar background
        pygame.draw.rect(self.screen, (220, 220, 220), (x, y, width, height), 0)
        
        # Calculate fill ratio
        ratio = np.clip(value / max_value, 0.0, 1.0)
        fill_color = color
        
        # Fill color changes based on performance (e.g., green for high HRV, red for high fatigue)
        if not is_positive and ratio > 0.8:
            fill_color = self.RED # High fatigue is bad
        elif is_positive and ratio < 0.5:
            fill_color = self.RED # Low HRV is bad
        
        pygame.draw.rect(self.screen, fill_color, (x, y, width * ratio, height), 0)
        
        # Outline
        pygame.draw.rect(self.screen, self.BLACK, (x, y, width, height), 2)
        
        # Value text
        self._draw_text(f"{value:.2f}", x + width + 10, y + 5, self.BLACK)

    def _draw_performance_pathway(self, x, y, hrv, fatigue):
        """Metaphorical visualization of athlete status on a grid."""
        GRID_SIZE = 50
        NUM_CELLS = 5
        
        self._draw_text("Performance Pathway (Recovery vs Fatigue)", x, y - 20, self.BLACK)

        # Draw 5x5 grid
        for i in range(NUM_CELLS):
            for j in range(NUM_CELLS):
                rect = pygame.Rect(x + i * GRID_SIZE, y + j * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)

        # Calculate agent position based on HRV (vertical) and Fatigue (horizontal)
        # Low Fatigue (Good) -> Left (Index 0)
        # High HRV (Good) -> Top (Index 0)
        
        fatigue_idx = int(np.clip(fatigue / 5.0 * NUM_CELLS, 0, NUM_CELLS - 1)) # 0-4
        hrv_idx = int(np.clip((1.0 - hrv) * NUM_CELLS, 0, NUM_CELLS - 1)) # 0-4 (Inverted: 1.0 HRV is index 0)
        
        agent_x = x + fatigue_idx * GRID_SIZE + GRID_SIZE // 2
        agent_y = y + hrv_idx * GRID_SIZE + GRID_SIZE // 2
        
        # Color the agent based on overall health
        if hrv > 0.8 and fatigue < 1.5:
            agent_color = self.GREEN  # Optimal Zone
        elif hrv < 0.4 or fatigue > 3.0:
            agent_color = self.RED    # Danger Zone
        else:
            agent_color = self.BLUE   # Training Zone
            
        pygame.draw.circle(self.screen, agent_color, (agent_x, agent_y), GRID_SIZE // 3)
        self._draw_text("Agent", agent_x - 20, agent_y + 15, self.BLACK)

    def close(self):
        """Closes the Pygame window."""
        if self.is_open:
            pygame.quit()
            self.is_open = False
            
# End of rendering.py