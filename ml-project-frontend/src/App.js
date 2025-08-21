import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import LeafCheck from "./leafcheck";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/leaf-check" element={<LeafCheck />} />
      </Routes>
    </Router>
  );
}

export default App;
