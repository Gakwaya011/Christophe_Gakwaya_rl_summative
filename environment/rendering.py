import pygame
import numpy as np
from typing import List, Tuple, Optional
import sys

class NutritionVisualizer:
    """
    Advanced visualization for Simplified Athlete Nutrition Environment
    """
    
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AI Athlete Nutrition Optimizer - Simplified")
        
        # Colors
        self.BG_COLOR = (15, 23, 42)
        self.PANEL_COLOR = (30, 41, 59)
        self.TEXT_COLOR = (226, 232, 240)
        self.ACCENT_COLOR = (59, 130, 246)
        
        self.HRV_COLOR = (34, 197, 94)
        self.FATIGUE_COLOR = (239, 68, 68)
        self.GLYCOGEN_COLOR = (251, 146, 60)
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        self.clock = pygame.time.Clock()
        
    def draw_rounded_rect(self, surface, rect, color, radius=8):
        pygame.draw.rect(surface, color, rect, border_radius=radius)
    
    def draw_gauge(self, x: int, y: int, width: int, height: int, 
                   value: float, max_value: float, color: Tuple, label: str):
        """Draw simplified gauge"""
        self.draw_rounded_rect(self.screen, (x, y, width, height), self.PANEL_COLOR, 8)
        
        # Label
        label_surf = self.font_small.render(label, True, self.TEXT_COLOR)
        self.screen.blit(label_surf, (x + 10, y + 10))
        
        # Value text
        value_surf = self.font_medium.render(f"{value:.1f}", True, color)
        self.screen.blit(value_surf, (x + 10, y + 35))
        
        # Gauge bar
        bar_x, bar_y = x + 10, y + height - 25
        bar_width, bar_height = width - 20, 12
        
        # Background bar
        pygame.draw.rect(self.screen, (50, 50, 60), (bar_x, bar_y, bar_width, bar_height), border_radius=6)
        
        # Value bar
        fill_width = min((value / max_value) * bar_width, bar_width)
        if fill_width > 0:
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_width, bar_height), border_radius=6)
    
    def draw_timeline(self, x: int, y: int, width: int, height: int, 
                     history: dict, current_day: int):
        """Draw historical progress"""
        self.draw_rounded_rect(self.screen, (x, y, width, height), self.PANEL_COLOR, 8)
        
        title_surf = self.font_medium.render("15-Day Progress", True, self.TEXT_COLOR)
        self.screen.blit(title_surf, (x + 10, y + 10))
        
        if not history['hrv']:
            return
        
        # Graph area
        graph_x, graph_y = x + 50, y + 50
        graph_width, graph_height = width - 70, height - 80
        
        # Draw metrics
        days = len(history['hrv'])
        if days > 1:
            x_scale = graph_width / 14
            
            # HRV line
            hrv_points = []
            for i, hrv in enumerate(history['hrv']):
                px = graph_x + i * x_scale
                py = graph_y + graph_height - (hrv / 100 * graph_height)
                hrv_points.append((px, py))
            if len(hrv_points) > 1:
                pygame.draw.lines(self.screen, self.HRV_COLOR, False, hrv_points, 3)
            
            # Fatigue line
            fatigue_points = []
            for i, fatigue in enumerate(history['fatigue']):
                px = graph_x + i * x_scale
                py = graph_y + graph_height - (fatigue / 100 * graph_height)
                fatigue_points.append((px, py))
            if len(fatigue_points) > 1:
                pygame.draw.lines(self.screen, self.FATIGUE_COLOR, False, fatigue_points, 2)
    
    def render(self, env_state: dict, action_taken: Optional[Tuple[float, float]] = None):
        """Main render method"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Clear screen
        self.screen.fill(self.BG_COLOR)
        
        # Title
        title = self.font_large.render("Simplified Nutrition Optimizer", True, self.TEXT_COLOR)
        self.screen.blit(title, (50, 20))
        
        day_text = self.font_medium.render(f"Day {env_state['current_day'] + 1}/15", True, self.ACCENT_COLOR)
        self.screen.blit(day_text, (50, 70))
        
        # Gauges
        gauge_width, gauge_height = 250, 100
        
        self.draw_gauge(50, 120, gauge_width, gauge_height, 
                       env_state['hrv'], 100, self.HRV_COLOR, "HRV (ms)")
        
        self.draw_gauge(320, 120, gauge_width, gauge_height,
                       env_state['fatigue'], 100, self.FATIGUE_COLOR, "Fatigue (%)")
        
        self.draw_gauge(590, 120, gauge_width, gauge_height,
                       env_state['glycogen'], 100, self.GLYCOGEN_COLOR, "Glycogen (%)")
        
        # Action info
        if action_taken:
            protein, carbs = action_taken
            action_text = self.font_medium.render(
                f"Action: Protein={protein:.1f}g, Carbs={carbs:.1f}g", 
                True, self.TEXT_COLOR
            )
            self.screen.blit(action_text, (50, 240))
        
        # Timeline
        self.draw_timeline(50, 280, 800, 200, env_state['history'], env_state['current_day'])
        
        # Performance summary
        summary_x, summary_y = 870, 120
        summary_width, summary_height = 300, 360
        
        self.draw_rounded_rect(self.screen, (summary_x, summary_y, summary_width, summary_height),
                              self.PANEL_COLOR, 8)
        
        summary_title = self.font_medium.render("Performance", True, self.TEXT_COLOR)
        self.screen.blit(summary_title, (summary_x + 20, summary_y + 20))
        
        if env_state['history']['hrv']:
            stats = [
                ("Current Day", f"{env_state['current_day'] + 1}", self.TEXT_COLOR),
                ("Avg HRV", f"{np.mean(env_state['history']['hrv']):.1f} ms", self.HRV_COLOR),
                ("Avg Fatigue", f"{np.mean(env_state['history']['fatigue']):.1f}%", self.FATIGUE_COLOR),
                ("Total Reward", f"{sum(env_state['history']['rewards']):.0f}", self.ACCENT_COLOR),
            ]
            
            for i, (label, value, color) in enumerate(stats):
                y_pos = summary_y + 60 + i * 50
                label_surf = self.font_small.render(label, True, (180, 180, 190))
                value_surf = self.font_medium.render(value, True, color)
                self.screen.blit(label_surf, (summary_x + 30, y_pos))
                self.screen.blit(value_surf, (summary_x + 30, y_pos + 25))
        
        pygame.display.flip()
        self.clock.tick(30)
    
    def close(self):
        pygame.quit()