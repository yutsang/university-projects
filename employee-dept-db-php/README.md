# Employee-Department-Projects-Database Web Application

## Description
A web-based interface for managing an Employee-Department-Projects relational database. The application allows users to insert, deactivate, and update projects, as well as generate project summary reports, all via a user-friendly web interface.

## Features
- **Insert a New Project:**
  - Enter project data (number, name, location, controlling department).
  - Controlling department is selected from a dynamic list of departments.
- **Deactivate an Existing Project:**
  - Select a project to deactivate; removes all employees from the project and deletes the project.
- **Update Worker Information for a Project:**
  - View and modify employees working on a project, update hours, remove or add employees.
- **Project Summary Report:**
  - Select one or more projects (or all) to view summary info: project details, controlling department, manager, employee count, total hours, and personnel cost.

## Getting Started

### Prerequisites
- PHP 7.x or higher
- MySQL/MariaDB database
- Web server (e.g., Apache, Nginx with PHP support)

### Setup Instructions
1. **Clone or download this repository.**
2. **Database Setup:**
   - Create a MySQL database and import your Employee-Department-Projects schema and data.
   - Update the database credentials in each PHP file (`mini-project-*.php`) as needed:
     ```php
     $host = "<your-db-host>";
     $usrname = "<your-db-username>";
     $usrpw = "<your-db-password>";
     $dbname = "<your-db-name>";
     ```
   - **Security Note:** For production, move credentials to a config file outside web root and do not commit to version control.
3. **Deploy Files:**
   - Place all PHP and HTML files in your web server's document root or a subdirectory.
4. **Access the Application:**
   - Open `mini-project.html` in your browser to access the main menu.

## Usage
- Use the main menu to navigate to each function (insert, deactivate, update, summary).
- Follow on-screen instructions for each operation.
- All actions are performed via web forms and buttons.

## Project Structure
- `mini-project.html` — Main menu and navigation.
- `mini-project-create.php` — Insert new project.
- `mini-project-deactivate.php` — Deactivate project and remove employees.
- `mini-project-update.php` — Update project workers and hours.
- `mini-project-summary.php` — Generate project summary reports.

## Security Notice
- **Do not commit real database credentials to version control.**
- For production, use environment variables or a config file for sensitive information.
- Always validate and sanitize user input to prevent SQL injection.

## License
Specify your license here (e.g., MIT, GPL, etc.).

## Contact & Contributions
- For questions or suggestions, open an issue or contact the project maintainer.
- Contributions are welcome! Please fork the repo and submit a pull request.
