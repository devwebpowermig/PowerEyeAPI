import { useState } from 'react';
import { navData } from '../../lib/navData'
import { FaArrowRight, FaArrowLeft } from 'react-icons/fa'
import Accordion from 'react-bootstrap/Accordion'

import styles from './PwSidebar.module.css';


export default function PwSidebar() {

    const [open, setOpen] = useState(false)

    function toggleOpen() {
        setOpen(!open)
    }

    return (

        <div className={open ? styles.sidenav : styles.sidenavClosed}>
            <button className={styles.menuBtn} onClick={toggleOpen}>
                {open ? <FaArrowRight /> : <FaArrowLeft />}
            </button>
            <Accordion>
                {navData.map(item => {
                    return <div key={item.id} className={open ? styles.sideItem : styles.sideItemClosed}>
                        {open ? '' : item.icon}
                        <span className={open ? styles.linkText : styles.linkTextClosed}>{

                            <Accordion.Item eventKey={item.id}>
                                <Accordion.Header>{item.icon}{item.text}</Accordion.Header>
                                <Accordion.Body>
                                    {item.component}
                                </Accordion.Body>
                            </Accordion.Item>
                        }
                        </span>
                    </div>
                })}
            </Accordion>
        </div>
    )
}