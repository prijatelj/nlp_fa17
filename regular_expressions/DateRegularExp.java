/**
 * A regular expression that identifies temporal events.
 * @author Derek S. Prijatelj
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class DateRegularExp{
    /**
     * Contains various global constant strings of regexs to find dates.
     * Although this format takes more memory, it is more easily manageable.
     */
    // General regex match cases, not date validating
    //private static final String dayStr = "\\p{Alpha}{6,9}";
    private static final String dayNum = "\\p{Digit}{1,2}";
    //private static final String monthStr = "\\p{Alpha}{4,9}";
    private static final String monthNum = dayNum;
    private static final String yearNum = "-?\\p{Digit}+";

    // Regexs that also validate the string is a date
    //*
    private static final String monthStr = "(" + String.join("\\b)|(\\b",
        "(\\b" + "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November",
        "December" + "\\b)"
    ) + ")";
    private static final String dayStr = "((?i)" + String.join("\\b)|(\\b",
        "(\\b" + "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday" + "\\b)"
    ) + ")";


    private static final String singleNum = String.join("\\b)|(\\b",
        "(\\b" + "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine" + "\\b)"
    );
    private static final String tensNum = String.join("\\b)|(\\b",
        "(\\b" + "ten",
        "((twen)|(thir)|(four)|(fif)|(six)|(seven)|(eigh)|(nine))ty" + "\\b)"
    );
    private static final String teensNum = String.join("\\b)|(\\b",
        "(\\b" + "eleven", "twleve",
        "((thir)|(four)|(fif)|(six)|(seven)|(eight)|(nine))teen" + "\\b)"
    );

    private static final String hours = "((?i)" + String.join("|",
        singleNum, teensNum,
        "(\\bten\\b)",
        "(\\btwenty-((one)|(two)|(three))\\b)"
    ) + ")";
    private static final String minutes= "((?i)" + String.join("|",
        singleNum, teensNum,
        "(\\bten\\b)",
        "((twen)|(thir)|(four)|(fif)|(six))ty" + "(-" + singleNum + ")?" +"\\b)"
    ) + ")";

    //*/
    private static final String holidays = "((?i)" + String.join("\\b)|(\\b",
        "(\\b" + "(((Islamic)|(Orthodox)|(Chinese))\\s)?"
            + "New\\sYear('s)?(\\sDay)?(\\sEve)?",
        "Tu\\sB'Shevat", "Arbor\\sDay",
        "Valentine's\\sDay",
        "Maha\\sShivaratri",
        "((Carnival)|(Shrove))\\sTuesday",
        "St.\\sDavid's\\sDay",
        "((Carnival)|(Ash))\\sWednesday",
        "Holi",
        "Purim",
        "St\\sPatrick's\\sDay",
        "((March)|(September))?\\sequinox",
        "Palm\\sSunday",
        "(Orthodox\\s)?Good\\sFriday",
        "(Orthodox\\s)?Holy\\sSaturday",
        "(Orthodox\\s)?Easter(\\s((Monday)|(Sunday)))?",
        "Shakespeare\\sDay",
        "Yom\\sHaShoah",
        "Isra\\sand\\sMi'raj",
        "St.\\sGeorge's\\sDay",
        "Yom\\sHaAtzmaut",
        "Shavuot",
        "Pentecost",
        "Father's\\sDay",
        "(((December)|(June))\\s)?Solstice",
        "Eid-al-Fitr",
        "Tisha\\sB'Av",
        "Raksha\\sBandhan",
        "Janmashtami",
        "Ganesh\\sChaturthi",
        "Eid-al-Adha",
        "Navaratri",
        "Rosh\\sHashana",
        "Muharram",
        "Dussehra",
        "Yom\\sKippur",
        "Diwali", "Deepavali",
        "Halloween", "All\\sSaints(')?\\sDay(\\sEve)?",
        "(Orthodox\\s)?Christmas(\\sDay)?(\\sEve)?", "Boxing\\sDay",
        "Hanukkah" + "\\b)"
    ) + ")";


        // Below regex is modified from the orignal regex from Varun Achar's
        // post on Stack Overflow at: https://stackoverflow.com/questions/51224/regular-expression-to-match-valid-dates#answer-8768241
    private static final String mmddyyyy =
        "\\b(?:(?:(?:0?[13578]|1[02])(?<PUNC>\\/|-|\\.)31)\\k<PUNC>|(?:(?:0?[1,3-9]|1[0-2])(?<PUNC2>\\/|-|\\.)(?:29|30)\\k<PUNC2>))(?:(?:1[6-9]|[2-9]\\d)?\\d+)\\b|\\b(?:0?2(?<PUNC3>\\/|-|\\.)29\\k<PUNC3>(?:(?:(?:1[6-9]|[2-9]\\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))\\b|\\b(?:(?:0?[1-9])|(?:1[0-2]))(?<PUNC4>\\/|-|\\.)(?:0?[1-9]|1\\d|2[0-8])\\k<PUNC4>(?:(?:1[6-9]|[2-9]\\d)?\\d+)\\b";
        //"\\b(?:(?:(?:0?[13578]|1[02])(\\/|-|\\.)31)\\1|(?:(?:0?[1,3-9]|1[0-2])(\\/|-|\\.)(?:29|30)\\2))(?:(?:1[6-9]|[2-9]\\d)?-?\\d+)\\b|\\b(?:0?2(\\/|-|\\.)29\\3(?:(?:(?:1[6-9]|[2-9]\\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))\\b|\\b(?:(?:0?[1-9])|(?:1[0-2]))(\\/|-|\\.)(?:0?[1-9]|1\\d|2[0-8])\\4(?:(?:1[6-9]|[2-9]\\d)?-?\\d+)\\b";

    // Patterns to observe
    private static final String absoluteDateRegex = String.join(")|(",
        "(\\b" + dayStr + ",\\s" + monthStr + "\\s" + dayNum + ",\\s" + yearNum
            + "\\b",
        "\\b" + monthStr + "\\s" + dayNum + "\\b",
        "\\b(the\\s)?" + dayNum + "((nd)|(st)|(th))?\\sof\\s" + monthStr + "\\b",
        holidays,
        mmddyyyy + ")"
    );
    private static final String diecticDateRegex = "((?i)"
        + String.join("\\b)|(\\b",
        "(\\b"+ "(yesterday)", "(tomorrow)", "(today)", "(yesteryear)",
        "(a|(1-9)+|" + singleNum + ")\\s((days?)|(weeks?)|(months?)|(years?))"
            + "\\sfrom\\s((yesterday)/(tomorrow)/(today))" + "\\b)"
    ) + ")";

    private static final String timeOfDayRegex = "((?i)"
        + String.join("\\b)|(\\b",
        "(\\b" + "([1-9]|([12][0-9]))\\so'clock",
        "[012]?[0-9][:\\.][0-5][0-9]\\s((p\\.?m\\.?)|(a\\.?m\\.?))?" + "\\b)"
    ) + ")";

    public static void main(String[] args){
        // read in file or input from terminal.
        // count all occurrences of checks in a file
        //True or False if terminal input matched by regex
        String filename = "";
        if (args.length >= 1){
            filename = args[0];
        } else {
            return;
        }

        int abs = 0;
        int diectic = 0;
        int timeOfDay = 0;

        Pattern absPattern = Pattern.compile(absoluteDateRegex);
        //Pattern absPattern = Pattern.compile(mmddyyyy);
        //Pattern absPattern = Pattern.compile(holidays);
        Pattern diecticPattern = Pattern.compile(diecticDateRegex);
        Pattern timeOfDayPattern = Pattern.compile(timeOfDayRegex);

        Matcher absMatcher = null;
        Matcher diecticMatcher = null;
        Matcher timeOfDayMatcher = null;

        System.out.println(absoluteDateRegex + "\n");
        System.out.println(diecticDateRegex + "\n");
        System.out.println(timeOfDayRegex + "\n");

        try{
            BufferedReader reader = new BufferedReader(
                new FileReader(filename));
            String line;
            while ((line = reader.readLine()) != null){
                absMatcher = absPattern.matcher(line);
                diecticMatcher = diecticPattern.matcher(line);
                timeOfDayMatcher = timeOfDayPattern.matcher(line);

                while (absMatcher.find()){
                    abs++;
                }
                while (diecticMatcher.find()){
                    diectic++;
                    System.out.println(diecticMatcher.group());
                }
                while (timeOfDayMatcher.find()){
                    timeOfDay++;
                    System.out.println(timeOfDayMatcher.group());
                }
            }
            reader.close();

            System.out.println("Absolute dates = " + abs);
            System.out.println("Diectic dates = " + diectic);
            System.out.println("Both dates = " + (abs + diectic));
            System.out.println("time-of-day expressions = " + timeOfDay);

        } catch (Exception e){
            System.err.format("Exception occurred trying to read '%s'.",
                filename);
            e.printStackTrace();
            return;
        }
    }
}
