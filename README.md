# BioSense

## This project was designed for the 2026 NASA Space Apps Hackathon

Project Overview
BioSense is a real-time health monitoring and simulation platform. It allows users to configuer a custom astronaut health simulation session. It can simulate various kinds of data, and uses machine learning algorithms to send anomaly alerts.

BioSense has the ability to send multiple different data signals using a single WebSocket connection. A user can simultaneously simulate and monitor an astronauts heart rate, blood sugar, brain waves, and blood oxygen/cabin CO2 levels.

When setting up the system, users can configure what statistical model they would like to use for each system. users can choose between static clinical thresholds, dynamic z-score anomalyd etection, exponential moving average drift detection, beta/alpha cognitive load calssifiers, and multivariate isolation forest machine learning models.

The front-end renders real-time 10-second sliding window telemtry charts using Chart.js, automatically highlighting AI-flagged anomalies in red.

There is a Master Console that uses a pub/sub architecture to track all active astronaut sessions, displaying live health statuses and alerts across the entire crew in real time.

Benefits & Intended Impact
This platform is able to catch subtle multivariate shifts, helping prevent medical emergencies before they escalate.

This platform gives students, researchers, and developers an instant zero-hardware sandbox to test how machine learning models respond to different types of data. Using this system, developers can create different models. When a model is ready, they can then connect real sensors and test out the system.

BioSense was designed to be lightweight, ensuring majority of systems can handle the setup and easily manage the data load. The database is connected directly to the WebSocket session state. If a connection ends for any reason, the data is removed to maintain the lightweight, cost-effective system.

Tools & Languages
Backend & AI
Python 3, FastAPI, Uvicorn, NumPy, and Scikit-learn

Frontend
HTML5, CSS3, JavaScript, WebSockets API, Chart.js

Database
Turso via libsql-client for low-latency session writes and automated teardown.

Cloud Hosting & DevOps
GitHub for CI/CD, Render Web Services for hosting the Python backend, Render Static Sites for hosting the data Simulation Console and Master Console.

What Makes the Project Creative
BioSense handles multiple independent streams of data and AI histories through a single WebSocket per session paired with a real-time Master Console that aggregates every active connection live.

Instead of just using generic data, environmental data is also being used. BioSense simulates Cabin CO2 levels with astronaut SpO2, and brainwaves with cognitive overload.

The database is self cleaning, eliminating storage bloat, and keeping the entire system much lighter and easier to maintain.

Key Factors
WebSockets were chosen over standard HTTP polling so that simulated signals and critical AI alerta travel bi-directionally without connection overhead.

Sliding history buffers are used to assist in accurate prediction and decision making for the machine learning algorithms. This allows the machine learning algorithm to become more effective without taking up too much space on the user interface.

Cloud resources are used to make this program lightweight and efficient. It is designed to comfortably run within standard server RAM limits and automatically cleans up the database to repeated testing would not exhaust storage quotas.

The frontend is decoupled from the backend and database. Each part of the system can scale or redeploy independently without breaking the workflows.
