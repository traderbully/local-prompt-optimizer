import { NavLink, Route, Routes } from "react-router-dom";
import TaskList from "./pages/TaskList";
import TaskDetail from "./pages/TaskDetail";
import RunViewer from "./pages/RunViewer";
import Comparison from "./pages/Comparison";
import Winner from "./pages/Winner";

export default function App() {
  return (
    <div className="h-full flex flex-col">
      <header className="border-b border-ink-700 bg-ink-800 px-4 py-3 flex items-center gap-6">
        <div className="font-semibold tracking-tight">
          <span className="text-accent-500">▸</span> Local Prompt Optimizer
        </div>
        <nav className="flex gap-4 text-sm">
          <NavLink
            to="/"
            className={({ isActive }) =>
              isActive ? "text-ink-50" : "text-ink-300 hover:text-ink-50"
            }
            end
          >
            Tasks
          </NavLink>
          <NavLink
            to="/runs"
            className={({ isActive }) =>
              isActive ? "text-ink-50" : "text-ink-300 hover:text-ink-50"
            }
          >
            Active runs
          </NavLink>
        </nav>
        <div className="ml-auto text-xs text-ink-400">v0.5.0 — Stage 5 UI</div>
      </header>
      <main className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<TaskList />} />
          <Route path="/runs" element={<TaskList showRuns />} />
          <Route path="/tasks/:name" element={<TaskDetail />} />
          <Route path="/tasks/:name/runs/:runId" element={<RunViewer />} />
          <Route path="/tasks/:name/comparison" element={<Comparison />} />
          <Route path="/tasks/:name/winner/:slug" element={<Winner />} />
        </Routes>
      </main>
    </div>
  );
}
