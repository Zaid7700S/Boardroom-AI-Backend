import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Define the steps and their durations
steps = [
    ("Establish a cross-functional task force", 14),
    ("Develop a comprehensive emergency recovery plan", 14),
    ("Trigger the company's disaster recovery plan", 14),
    ("Issue a prompt public statement", 14)
]

# Define the start date
start_date = datetime(2024, 1, 1)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set the x-axis to dates
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Plot each step
for i, (step, duration) in enumerate(steps):
    end_date = start_date + timedelta(days=duration)
    ax.barh(i, duration, left=start_date, height=0.5, color=plt.cm.tab20(i), label=step)

# Set the y-axis ticks
ax.set_yticks(range(len(steps)))
ax.set_yticklabels([step for step, _ in steps])

# Set the title and labels
ax.set_title('Recovery Strategy Timeline')
ax.set_xlabel('Days')
ax.set_ylabel('Step')

# Rotate the x-axis labels
plt.gcf().autofmt_xdate()

# Legend
plt.legend(loc='upper right')

# Save the figure
plt.savefig('strategy_chart.png', bbox_inches='tight')
plt.show()