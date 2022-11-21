import * as React from "react";

import {
    Box,
    Slider,
    Button,
    Container,
    Card,
    CardContent,
    Typography,
    FormControl,
    FormLabel,
    RadioGroup,
    Radio,
    FormControlLabel,
    Switch,
} from "@mui/material";
import {Line, Bar} from "react-chartjs-2";
import {faker} from "@faker-js/faker";

import "./DataVisualization.css";

import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    BarElement,
} from "chart.js";

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    BarElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

export const DataVisualization = ({rawData}) => {
    const [sliderValue, setSliderValue] = React.useState(null);
    const [windowValue, setWindowValue] = React.useState(50);
    const [episodeLength, setEpisodeLength] = React.useState(0);
    const [priceGraph, setPriceGraph] = React.useState(null);
    const [custBehaviourGraph, setCustBehaviourGraph] = React.useState(null);
    const [profitGraph, setProfitGraph] = React.useState(null);
    const [waitingGraph, setWaitingGraph] = React.useState(null);
    const [incomingGraph, setIncomingGraph] = React.useState(null);
    const [activeVendor, setActiveVendor] = React.useState(0);
    const [accumulated, setAccumulated] = React.useState(false);
    const [avgPrices, setAvgPrices] = React.useState([]);

    React.useEffect(() => {
        if (!rawData) return;
        setSliderValue(0);
        setEpisodeLength(rawData["price_new"][0].length);
    }, [rawData]);

    React.useEffect(() => {
        if (!rawData) return;

        let data = JSON.parse(JSON.stringify(rawData["profits"]));

        if (accumulated) {
            rawData["profits"].forEach((vendor, idx) => {

                const cumulativeSum = (
                    (sum) => (value) =>
                        (sum += value)
                )(0);
                data[idx] = data[idx].map(cumulativeSum);
            });
        }

        setPriceGraph(
            getPriceData(sliderValue, episodeLength, rawData, activeVendor, windowValue)
        );
        setCustBehaviourGraph(
            getCustomerBehaviour(sliderValue, episodeLength, rawData, activeVendor)
        );
        setProfitGraph(
            getProfitGraph(sliderValue, episodeLength, data, activeVendor, windowValue)
        );

        if (rawData['customers_waiting']) {
            setWaitingGraph(getWatingGraph(sliderValue, episodeLength, rawData, activeVendor));
        }

        if (rawData['incoming_customer']) {
            setIncomingGraph(getIncomingGraph(sliderValue, episodeLength, rawData, activeVendor));
        }


        if (rawData['is_linear']) {

            const avg_price = [];

            for (let vendor = 0; vendor < rawData["vendors"]; vendor++) {
                const prices = rawData['price_new'][vendor];
                const sales = rawData['sales_new'][vendor];
                const avg = prices.map((price, idx) => price * sales[idx]).reduce((a, b) => a + b, 0) / sales.reduce((a, b) => a + b, 0)
                avg_price.push(avg);
            }

            setAvgPrices(avg_price);

        } else {
            // TODO
        }

    }, [sliderValue, activeVendor, accumulated, windowValue]);

    const handleRadioChange = (event) => {
        setActiveVendor(parseInt(event.target.value));
    };

    if (rawData === null) {
        return (
            <Box
                sx={{
                    display: "flex",
                    height: "100%",
                    width: "100%",
                    justifyContent: "center",
                    alignItems: "center",
                }}
            >
                Please start by uploading data.
            </Box>
        );
    }

    return (
        <Box
            sx={{
                display: "flex",
                height: "100%",
                width: "100%",
                flexDirection: "column",
            }}
        >
            <Box
                sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                }}
            >
                <Box className={"statistics_boxes"}>
                    <Card>
                        <CardContent>
                            <Typography variant="h5">
                                # vendors: {rawData["vendors"]}
                            </Typography>
                        </CardContent>
                    </Card>
                    <Card>
                        <CardContent>
                            <Typography variant="h5">
                                episode length: {episodeLength}
                            </Typography>
                        </CardContent>
                    </Card>
                    <p></p>
                </Box>
                <Box
                    sx={{
                        display: "flex",
                        flexDirection: "column",
                    }}
                >
                    <Box sx={{width: 500, marginBottom: "30px", display: "flex", alignItems: "center"}}>
                        <h3>time step</h3>
                        <Slider
                            value={sliderValue}
                            onChange={(e) => setSliderValue(e.target.value)}
                            step={1}
                            valueLabelDisplay="on"
                            marks
                            min={0}
                            max={episodeLength - 1}
                        />
                    </Box>
                    <Box sx={{width: 500,  display: "flex", alignItems: "center"}}>
                        <h3>window length</h3>
                        <Slider
                            value={windowValue}
                            onChange={(e) => setWindowValue(e.target.value)}
                            step={1}
                            valueLabelDisplay="on"
                            marks
                            min={5}
                            max={episodeLength - 1}
                        />
                    </Box>
                </Box>

            </Box>
            {rawData && rawData["vendors"] > 1 && (
                <FormControl
                    sx={{
                        display: "flex",
                        flexDirection: "row",
                        alignItems: "center",
                        marginTop: "1rem",
                    }}
                >
                    <FormLabel sx={{marginRight: "1rem"}}>Vendor</FormLabel>
                    <RadioGroup
                        value={activeVendor}
                        onChange={handleRadioChange}
                        row={true}
                    >
                        {" "}
                        {Array.from(Array(rawData["vendors"]).keys()).map((i) => {
                            return (
                                <FormControlLabel
                                    value={i}
                                    control={<Radio/>}
                                    label={`Vendor ${i}`}
                                />
                            );
                        })}
                        <FormControlLabel value={-1} control={<Radio/>} label={`All`}/>
                    </RadioGroup>
                </FormControl>
            )}

            <Box sx={{display: "flex", marginTop: "2rem"}}>
                {priceGraph && (
                    <Box sx={{flexGrow: "1"}}>
                        <Line
                            options={{
                                responsive: true,
                                plugins: {
                                    legend: {
                                        position: "top",
                                    },
                                    title: {
                                        display: true,
                                        text: "Price Behaviour",
                                    },
                                },
                                scales: {
                                    y: {
                                        max: 10,
                                        suggestedMin: 2,
                                    },
                                },
                            }}
                            data={priceGraph}
                        />
                    </Box>
                )}

                {custBehaviourGraph && (
                    <Box sx={{height: "100%", width: "200px", display: "flex"}}>
                        <Bar
                            options={{
                                responsive: true,
                                maintainAspectRatio: false,
                                scales: {
                                    x: {
                                        stacked: true,
                                    },
                                    y: {
                                        stacked: true,
                                        max: 20,
                                    },
                                },
                                plugins: {
                                    legend: {
                                        position: "top",
                                    },
                                    title: {
                                        display: true,
                                        text: "Customer Behaviour",
                                    },
                                },
                            }}
                            data={custBehaviourGraph}
                        />
                    </Box>
                )}
            </Box>

            <Box sx={{marginTop: "3rem"}} className={"statistics_boxes"}>
                {!rawData["is_linear"] && (<Card>
                    <CardContent>
                        <Typography variant="h5">
                            <strong>in circulation</strong>:{" "}
                            {rawData["in_circulation"][sliderValue]}
                        </Typography>
                    </CardContent>
                </Card>)}
                {activeVendor !== -1 && !rawData["is_linear"] && (
                    <Card>
                        <CardContent>
                            <Typography variant="h5">
                                <strong>storage vendor {activeVendor}</strong>:{" "}
                                {rawData["in_storage"][activeVendor][sliderValue]} /{" "}
                                {rawData["max_storage"]}
                            </Typography>
                        </CardContent>
                    </Card>
                )}
                {activeVendor === -1 && !rawData["is_linear"] &&
                    Array.from(Array(rawData["vendors"]).keys()).map((i) => {
                        return (
                            <>
                                <Card>
                                    <CardContent>
                                        <Typography variant="h5">
                                            <strong>storage vendor {i}</strong>:{" "}
                                            {rawData["in_storage"][i][sliderValue]} /{" "}
                                            {rawData["max_storage"]}
                                        </Typography>
                                    </CardContent>
                                </Card>
                            </>
                        );
                    })}
                {activeVendor !== -1 && (
                    <Card>
                        <CardContent>
                            <Typography variant="h5">
                                <strong>total profit vendor {activeVendor}</strong>:{" "}
                                {Number(
                                    rawData["profits"][activeVendor].reduce(
                                        (pv, cv) => pv + cv,
                                        0
                                    )
                                ).toFixed(2)}
                            </Typography>
                        </CardContent>
                    </Card>
                )}
                {activeVendor === -1 &&
                    Array.from(Array(rawData["vendors"]).keys()).map((i) => {
                        return (
                            <>
                                <Card>
                                    <CardContent>
                                        <Typography variant="h5">
                                            <strong>total profit vendor {i}</strong>:{" "}
                                            {Number(
                                                rawData["profits"][i].reduce((pv, cv) => pv + cv, 0)
                                            ).toFixed(2)}
                                        </Typography>
                                    </CardContent>
                                </Card>
                            </>
                        );
                    })}
                {rawData["is_linear"] && activeVendor === -1 && Array.from(Array(rawData["vendors"]).keys()).map((i) => (
                    <Card key={i}>
                        <CardContent>
                            <Typography variant="h5">
                                <strong>Ø new buy price - vendor {i}</strong>:{" "}
                                {Number(
                                    avgPrices[i]
                                ).toFixed(2)}
                            </Typography>
                        </CardContent>
                    </Card>))}
                {rawData["is_linear"] && activeVendor !== -1 && <Card>
                    <CardContent>
                        <Typography variant="h5">
                            <strong>Ø new buy price - vendor {activeVendor}</strong>:{" "}
                            {Number(
                                avgPrices[activeVendor]
                            ).toFixed(2)}
                        </Typography>
                    </CardContent>
                </Card>}
            </Box>
            <Box
                sx={{
                    display: "flex",
                    justifyContent: "flex-end",
                    alignItems: "center",
                    marginTop: "2rem",
                }}
            >
                <FormControlLabel
                    control={
                        <Switch
                            label="accumulated"
                            onChange={(e) => {
                                console.log(e);
                                setAccumulated(e.target.checked);
                            }}
                            value={accumulated}
                        />
                    }
                    label="accumulated"
                />
            </Box>
            <Box sx={{display: "flex", marginBottom: "2rem"}}>
                {profitGraph && (
                    <Box sx={{flexGrow: "1"}}>
                        <Line
                            options={{
                                responsive: true,
                                plugins: {
                                    legend: {
                                        position: "top",
                                    },
                                    title: {
                                        display: true,
                                        text: "Profit",
                                    },
                                },
                                scales: {
                                    y: {
                                        suggestedMin: 0,
                                    },
                                },
                            }}
                            data={profitGraph}
                        />
                    </Box>
                )}
            </Box>
            <Box sx={{display: "flex", marginBottom: "2rem"}}>
                {waitingGraph && (
                    <Box sx={{flexGrow: "1"}}>
                        <Line
                            options={{
                                responsive: true,
                                plugins: {
                                    legend: {
                                        position: "top",
                                    },
                                    title: {
                                        display: true,
                                        text: "Waiting Customers",
                                    },
                                },
                                scales: {
                                    y: {
                                        suggestedMin: 0,
                                    },
                                    x: {
                                        ticks: {
                                            maxTicksLimit: 10
                                        }
                                    }
                                },
                            }}
                            data={waitingGraph}
                        />
                    </Box>
                )}
            </Box>
            <Box sx={{display: "flex", marginBottom: "2rem"}}>
                {incomingGraph && (
                    <Box sx={{flexGrow: "1"}}>
                        <Line
                            options={{
                                responsive: true,
                                plugins: {
                                    legend: {
                                        position: "top",
                                    },
                                    title: {
                                        display: true,
                                        text: "Incoming Customers",
                                    },
                                },
                                scales: {
                                    y: {
                                        suggestedMin: 0,
                                    },
                                    x: {
                                        ticks: {
                                            maxTicksLimit: 10
                                        }
                                    }
                                },
                            }}
                            data={incomingGraph}
                        />
                    </Box>
                )}
            </Box>
            <Box sx={{width: 500}}>
                <Slider
                    aria-label="Temperature"
                    value={sliderValue}
                    onChange={(e) => setSliderValue(e.target.value)}
                    step={1}
                    valueLabelDisplay="on"
                    marks
                    min={0}
                    max={episodeLength - 1}
                />
            </Box>
        </Box>
    );
};

const getBounds = (slidingPos, episodeLength, window = 100) => {

    let lowerBound = slidingPos;
    let upperBound = slidingPos + window;

    if (upperBound > episodeLength) {
        lowerBound = episodeLength - window;
        upperBound = episodeLength;
    }

    return {lowerBound, upperBound};
};

const offsetData = (data, index = 0) => {
    const newData = [];
    data.forEach(datapoint => {
        if (index === 0) {
            newData.push(datapoint, null)
        } else {
            newData.push(null, datapoint)
        }
    })
    return newData
}

export const getPriceData = (slidingPos, episodeLength, data, vendor, windowLength) => {
    const {lowerBound, upperBound} = getBounds(slidingPos, episodeLength, windowLength);

    const labels = [];

    for (let i = lowerBound; i < upperBound; i++) {
        labels.push(`timestep ${i}`);
        labels.push("")
    }

    const isCircular = !data["is_linear"];

    if (vendor !== -1) {
        const datasets = [
            {
                label: "Price New",
                data: offsetData(data["price_new"][vendor].slice(lowerBound, upperBound), vendor),
                stepped: true,
                spanGaps: true,
                borderColor: "red",
            },
        ];

        if (isCircular) {
            datasets.push(
                {
                    label: "Price Refurbished",
                    data: offsetData(data["price_refurbished"][vendor].slice(lowerBound, upperBound), vendor),
                    stepped: true,
                    spanGaps: true,
                    borderColor: "green",
                },
                {
                    label: "Price Rebuy",
                    data: offsetData(data["price_rebuy"][vendor].slice(lowerBound, upperBound), vendor),
                    stepped: true,
                    spanGaps: true,
                    borderColor: "blue",
                }
            );
        }
        return {
            labels: labels,
            datasets: datasets,
        };
    }


    const datasets = [
        {
            label: "Price New from vendor 0",
            data: offsetData(data["price_new"][0].slice(lowerBound, upperBound)),
            stepped: true,
            spanGaps: true,
            borderColor: "red",
        },
        {
            label: "Price New from vendor 1",
            data: offsetData(data["price_new"][1].slice(lowerBound, upperBound), 1),
            stepped: true,
            spanGaps: true,
            borderColor: "purple",
        },
    ];

    if (isCircular) {
        datasets.push(
            {
                label: "Price Refurbished from vendor 0",
                data: offsetData(data["price_refurbished"][0].slice(lowerBound, upperBound)),
                stepped: true,
                spanGaps: true,
                borderColor: "green",
            },
            {
                label: "Price Rebuy from vendor 0",
                data: offsetData(data["price_rebuy"][0].slice(lowerBound, upperBound)),
                stepped: true,
                spanGaps: true,
                borderColor: "blue",
            },

            {
                label: "Price Refurbished from vendor 1",
                data: offsetData(data["price_refurbished"][1].slice(lowerBound, upperBound)),
                stepped: true,
                spanGaps: true,
                borderColor: "lightgreen",
            },
            {
                label: "Price Rebuy  from vendor 1",
                data: offsetData(data["price_rebuy"][1].slice(lowerBound, upperBound)),
                stepped: true,
                spanGaps: true,
                borderColor: "lightblue",
            }
        );
    }

    // only support comp = 2
    return {
        labels: labels,
        datasets: datasets,
    };
};

export const getWatingGraph = (slidingPos,
                               episodeLength,
                               data,
                               vendor) => {

    const datasets = [];
    const lowerBound = 0;
    const upperBound = data['customers_waiting'].length
    const labels = [];

    for (let i = lowerBound; i < upperBound; i++) {
        labels.push(`timestep ${i}`);
    }

    datasets.push({
        label: "waiting customer",
        data: data["customers_waiting"].slice(lowerBound, upperBound),
        backgroundColor: "green",
    });

    return {
        labels: labels,
        datasets: datasets,
    };

}

export const getIncomingGraph = (slidingPos,
                                 episodeLength,
                                 data,
                                 vendor) => {

    const datasets = [];
    const lowerBound = 0;
    const upperBound = data['incoming_customer'].length
    const labels = [];

    for (let i = lowerBound; i < upperBound; i++) {
        labels.push(`timestep ${i}`);
    }

    datasets.push({
        label: "incoming customer",
        data: data["incoming_customer"],
        backgroundColor: "green",
    });

    return {
        labels: labels,
        datasets: datasets,
    };

}

export const getCustomerBehaviour = (
    slidingPos,
    episodeLength,
    data,
    vendor
) => {
    const isCircular = !data["is_linear"];

    if (vendor !== -1) {
        const datasets = [
            {
                label: "buy new",
                data: [data["sales_new"][vendor][slidingPos]],
                backgroundColor: "red",
            },
            {
                label: "no buy",
                data: [data["sales_no_buy"][slidingPos]],
                backgroundColor: "black",
            },
        ];

        if (isCircular) {
            datasets.push({
                label: "buy refurbished",
                data: [data["sales_refurbished"][vendor][slidingPos]],
                backgroundColor: "green",
            });
        }
        return {
            labels: [`timestep ${slidingPos}`],
            datasets: datasets,
        };
    }

    const datasets = [
        {
            label: "buy new from 0",
            data: [data["sales_new"][0][slidingPos]],
            backgroundColor: "red",
        },
        {
            label: "no buy",
            data: [data["sales_no_buy"][slidingPos]],
            backgroundColor: "black",
        },
        {
            label: "buy new from 1",
            data: [data["sales_new"][1][slidingPos]],
            backgroundColor: "purple",
        },
    ];

    if (isCircular) {
        datasets.push(
            {
                label: "buy refurbished from 0",
                data: [data["sales_refurbished"][0][slidingPos]],
                backgroundColor: "green",
            },
            {
                label: "buy refurbished from 1",
                data: [data["sales_refurbished"][1][slidingPos]],
                backgroundColor: "lightgreen",
            }
        );
    }

    return {
        labels: [`timestep ${slidingPos}`],
        datasets: datasets,
    };
};

export const getProfitGraph = (slidingPos, episodeLength, data, vendor, windowLength) => {

    const {lowerBound, upperBound} = getBounds(slidingPos, episodeLength, windowLength);

    const labels = [];

    for (let i = lowerBound; i < upperBound; i++) {
        labels.push(`timestep ${i}`);
    }

    if (vendor !== -1)
        return {
            labels: labels,
            datasets: [
                {
                    label: "Profit",
                    data: data[vendor].slice(lowerBound, upperBound),
                    stepped: true,
                    borderColor: "red",
                },
            ],
        };

    // only support comp = 2
    return {
        labels: labels,
        datasets: [
            {
                label: "Profit vendor 0",
                data: data[0].slice(lowerBound, upperBound),
                stepped: true,
                borderColor: "red",
            },
            {
                label: "Profit vendor 1",
                data: data[1].slice(lowerBound, upperBound),
                stepped: true,
                borderColor: "purple",
            },
        ],
    };
};
