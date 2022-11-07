import axios from 'axios';
import PowerEye from './components/PowerEye';
import PwNavbar from './navigation/Navbar/Navbar';
import PwSidebar from './navigation/Sidebar/PwSidebar';
import './App.scss';
const POWEREYE = 'http://127.0.0.1:8000/powereye'

function App() {


  async function loadPowerEye() {
    try {
      await axios.get(POWEREYE)
    } catch (err) {
      console.error(err);
    }
  }

  return (
    <div className="App">
      <PwNavbar />
      <main>
        <button onClick={loadPowerEye}>Rodar PowerEye</button>

      </main>
      <PwSidebar />
    </div >

  );
}

export default App;
