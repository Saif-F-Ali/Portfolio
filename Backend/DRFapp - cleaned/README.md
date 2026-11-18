
# 🎬 Movie Watchlist & Review API

A RESTful API built with **Django** and **Django REST Framework (DRF)** for managing a movie watchlist, streaming platforms, and user reviews. 

This project was built as a backend portfolio piece to demonstrate practical experience with API design, authentication, permissions, and test-driven development (TDD).

## ✨ Features

*   **Watchlist Management:** Browse paginated movie entries containing storylines, streaming platforms, and dynamically calculated average ratings.
*   **Streaming Platforms:** Track which platform (e.g., Netflix, Hulu) hosts each movie.
*   **Role-Based Access Control:** 
    *   **Staff/Admin users** can create, update, and delete watchlist entries and streaming platforms.
    *   **Regular users** have read-only access to movies and platforms.
*   **Review System:**
    *   Registered users can create, update, and delete their own reviews.
    *   Users are restricted to **one review per movie** (validated at the serializer level).
    *   New reviews automatically update the movie's overall `avg_rating`.
    *   Filter and search reviews by username.
*   **Authentication:** Secure user registration, login, and logout using DRF's Token Authentication.

## 🛠️ Tech Stack

*   **Framework:** Django, Django REST Framework (DRF)
*   **Authentication:** DRF Token Authentication *(Note: JWT was explored and integrated for learning purposes, though Token Auth is used as the primary mechanism).*
*   **Database:** SQLite (Development) / MySQL (Planned for Deployment)
*   **Filtering:** `django-filter` for advanced querying and ordering.
*   **Testing:** DRF's `APITestCase` & Thunder Client (for manual endpoint testing).

## 🧪 Testing & Quality Assurance

This project features a robust test suite with **3 test classes and 20 test methods** to ensure API reliability, permission enforcement, and data integrity:

1.  **`WatchlistTestCase`:** Verifies CRUD operations for movies. Ensures regular users receive `403 Forbidden` when attempting to create/update/delete, while admin users can perform these actions successfully.
2.  **`StreamPlatformTestCase`:** Tests access control for streaming platforms, confirming that regular authenticated users can view platforms but cannot modify them.
3.  **`ReviewTestCase`:** Focuses on the review workflow. Tests include:
    *   Preventing duplicate reviews for the same movie.
    *   Blocking unauthenticated users from posting.
    *   Ensuring users can update/delete their own reviews.
    *   Fetching reviews by username.

## 🚀 Local Setup & Installation

To run this project locally, follow these steps:

**1. Clone the repository:**
```bash
git clone https://github.com/Saif-F-Ali/Portfolio/Backend.git
cd your-repo-name
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run database migrations:**
```bash
python manage.py migrate
```

**5. Start the development server:**
```bash
python manage.py runserver
```

The API will be available locally at http://127.0.0.1:8000/.

Note: Because this is a pure backend API, visiting the root URL directly in a browser will return a 404.
To interact with the endpoints, use an API client like Postman, Thunder Client, or the DRF Browsable API (by appending specific routes like /api/watch/ to the base URL).

## 📡 API Endpoints Overview


| Method | Endpoint | Description | Access |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/account/register/` | Register a new user | Public |
| `POST` | `/api/account/login/` | Login and receive a token | Public |
| `GET` | `/api/watch/platforms/` | List all platforms | Authenticated |
| `POST` | `/api/watch/platforms/` | Create a new platform | Admin/Staff only |
| `GET` | `/api/watch/` | List all movies (paginated) | Authenticated |
| `POST` | `/api/watch/` | Create a new movie | Admin/Staff only |
| `GET` | `/api/watch/{id}/` | Retrieve a specific movie | Authenticated |
| `GET` | `/api/watch/{id}/revs/` | List all reviews for a movie | Authenticated |
| `POST` | `/api/watch/{id}/revs/create/` | Add a review for a movie | Authenticated |
| `GET` | `/api/watch/user-revs/?username=` | List all reviews for a reviewer | Authenticated |

## 🧠 Key Learnings & Challenges

*   **Custom Serializer Logic:** Overrode the default `.create()` and `.update()` methods in serializers to handle nested Foreign Key relationships (e.g., dynamically linking or creating streaming platforms).
*   **Validation:** Implemented custom field-level validation to prevent short titles and duplicate reviews while ensuring updates to existing objects do not trigger false positives.
*   **Dynamic Fields:** Utilized `SerializerMethodField` to calculate and inject custom data into the JSON response (like average ratings and warning flags) without altering the underlying database schema.

---
*Developed by **Saif F. Ali** as a portfolio project for backend engineering roles.*
