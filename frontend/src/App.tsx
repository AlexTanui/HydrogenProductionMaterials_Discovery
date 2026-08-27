import { useEffect, useState } from "react";
import { NavLink, Route, HashRouter as Router, Routes } from "react-router-dom";
import { IconChart, IconFlask, IconHome, IconMoon, IconSun } from "./components/icons";
import Benchmarks from "./pages/Benchmarks";
import Dashboard from "./pages/Dashboard";
import Predict from "./pages/Predict";

type Theme = "light" | "dark";

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = localStorage.getItem("theme");
    if (stored === "light" || stored === "dark") return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  return [theme, () => setTheme((t) => (t === "light" ? "dark" : "light"))];
}

function navClass({ isActive }: { isActive: boolean }) {
  return isActive ? "nav-link active" : "nav-link";
}

export default function App() {
  const [theme, toggleTheme] = useTheme();

  return (
    <Router>
      <div className="app-shell">
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-mark" />
            <div>
              <div className="brand-text">Hydrogen Discovery</div>
              <div className="brand-subtext">PG-S2-47</div>
            </div>
          </div>

          <nav className="nav">
            <NavLink to="/" end className={navClass}>
              <IconHome /> Dashboard
            </NavLink>
            <NavLink to="/predict" className={navClass}>
              <IconFlask /> Predict
            </NavLink>
            <NavLink to="/benchmarks" className={navClass}>
              <IconChart /> Benchmarks
            </NavLink>
          </nav>

          <div className="sidebar-footer">
            <button className="theme-toggle" onClick={toggleTheme}>
              {theme === "light" ? <IconMoon /> : <IconSun />}
              {theme === "light" ? "Dark mode" : "Light mode"}
            </button>
          </div>
        </aside>

        <main className="main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/benchmarks" element={<Benchmarks />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
