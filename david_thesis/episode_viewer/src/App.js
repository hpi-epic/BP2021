import * as React from 'react';

import './App.css';
import CssBaseline from '@mui/material/CssBaseline';
import {Box, Button, Container} from '@mui/material';

import {DataVisualization} from "./DataVisualization";

function App() {

    const [rawData, setRawData] = React.useState(null);


    const handleChange = (event) => {
        if (event.target.files[0]) {
            const fileReader = new FileReader();
            fileReader.readAsText(event.target.files[0], "UTF-8");
            fileReader.onload = e => {
                console.log("e.target.result", e.target.result);
                setRawData(JSON.parse(e.target.result));
            };
        }

    }

    return (
        <div className="App">
            <CssBaseline/>
            <Container fixed sx={{height: "100vh", display: "flex", flexDirection: "column"}}>
                <Box sx={{display: 'flex', justifyContent: "space-between", alignItems: "center"}}>
                    <h2>Episode Viewer</h2>
                    <Button
                        variant="contained"
                        sx={{height: "3rem"}}
                        component="label"
                    >
                        Upload File
                        <input
                            type="file"
                            onChange={handleChange}
                            accept={"application/JSON"}
                            hidden
                        />
                    </Button>
                </Box>
                <Box sx={{display: 'flex', justifyContent: "center", alignItems: "center", flexGrow: 1}}>
                    <DataVisualization rawData={rawData}/>
                </Box>
            </Container>
        </div>
    )
        ;
}

export default App;
